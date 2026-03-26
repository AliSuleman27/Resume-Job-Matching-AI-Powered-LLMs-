import os
import logging
import uuid
import datetime
from flask import current_app
from bson.objectid import ObjectId
from extensions import recruiters_collection

logger = logging.getLogger(__name__)

# Allow OAuth over HTTP for local development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# ---------------------------------------------------------------------------
# IMPORTANT: GCP OAuth Consent Screen — "Testing" vs "Production" mode
# ---------------------------------------------------------------------------
# If only manually-added test users can authenticate (403 access_denied for
# everyone else), the OAuth consent screen is still in **Testing** mode.
#
# To fix:
#   1. Go to GCP Console → APIs & Services → OAuth consent screen
#   2. Click "PUBLISH APP" to move from Testing → Production
#   3. Google may require a verification review for sensitive scopes
#      (e.g. calendar access). Follow the prompts and submit.
#
# This is NOT a code issue — it is a GCP Console configuration step.
# ---------------------------------------------------------------------------

SCOPES = ['https://www.googleapis.com/auth/calendar']


def google_calendar_configured():
    """Check if Google Calendar OAuth env vars are set."""
    return bool(
        current_app.config.get('GOOGLE_CLIENT_ID')
        and current_app.config.get('GOOGLE_CLIENT_SECRET')
    )


def _make_flow():
    """Create a reusable Flow instance from app config."""
    from google_auth_oauthlib.flow import Flow

    return Flow.from_client_config(
        {
            'web': {
                'client_id': current_app.config['GOOGLE_CLIENT_ID'],
                'client_secret': current_app.config['GOOGLE_CLIENT_SECRET'],
                'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                'token_uri': 'https://oauth2.googleapis.com/token',
            }
        },
        scopes=SCOPES,
        redirect_uri=current_app.config['GOOGLE_REDIRECT_URI'],
    )


def build_auth_url():
    """Generate Google OAuth consent URL. Returns (auth_url, code_verifier)."""
    flow = _make_flow()
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        prompt='consent',
        include_granted_scopes='true',
    )
    # PKCE: flow.code_verifier is set by authorization_url()
    return auth_url, getattr(flow, 'code_verifier', None)


def exchange_code_for_tokens(url, code_verifier=None):
    """Exchange the OAuth callback URL for tokens."""
    flow = _make_flow()
    if code_verifier:
        flow.code_verifier = code_verifier
    flow.fetch_token(authorization_response=url)
    credentials = flow.credentials
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': list(credentials.scopes) if credentials.scopes else SCOPES,
        'expiry': credentials.expiry.isoformat() if credentials.expiry else None,
    }


def store_google_tokens(recruiter_id, token_data):
    """Save Google OAuth tokens to recruiter document."""
    token_data['connected_at'] = datetime.datetime.now(datetime.timezone.utc)
    recruiters_collection.update_one(
        {'_id': ObjectId(recruiter_id)},
        {'$set': {'google_tokens': token_data}},
    )


def get_google_credentials(recruiter_id):
    """Load Google credentials from MongoDB, auto-refresh if expired."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    recruiter = recruiters_collection.find_one(
        {'_id': ObjectId(recruiter_id)},
        {'google_tokens': 1},
    )
    if not recruiter or not recruiter.get('google_tokens'):
        return None

    tokens = recruiter['google_tokens']
    creds = Credentials(
        token=tokens.get('token'),
        refresh_token=tokens.get('refresh_token'),
        token_uri=tokens.get('token_uri', 'https://oauth2.googleapis.com/token'),
        client_id=tokens.get('client_id', current_app.config.get('GOOGLE_CLIENT_ID')),
        client_secret=tokens.get('client_secret', current_app.config.get('GOOGLE_CLIENT_SECRET')),
        scopes=tokens.get('scopes', SCOPES),
    )

    if creds.expiry:
        if isinstance(creds.expiry, str):
            creds.expiry = datetime.datetime.fromisoformat(creds.expiry)

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            recruiters_collection.update_one(
                {'_id': ObjectId(recruiter_id)},
                {'$set': {
                    'google_tokens.token': creds.token,
                    'google_tokens.expiry': creds.expiry.isoformat() if creds.expiry else None,
                }},
            )
        except Exception as e:
            logger.error(f"Failed to refresh Google token for recruiter {recruiter_id}: {e}")
            return None

    return creds


def recruiter_has_google(recruiter_id):
    """Quick check if recruiter has Google tokens stored."""
    recruiter = recruiters_collection.find_one(
        {'_id': ObjectId(recruiter_id)},
        {'google_tokens': 1},
    )
    return bool(recruiter and recruiter.get('google_tokens'))


def disconnect_google(recruiter_id):
    """Remove Google tokens from recruiter document."""
    recruiters_collection.update_one(
        {'_id': ObjectId(recruiter_id)},
        {'$unset': {'google_tokens': ''}},
    )


def create_calendar_event(recruiter_id, summary, description, start_dt, duration_minutes, timezone_str, attendee_emails):
    """Create a Google Calendar event with a Google Meet link.

    Returns dict with event_id, meet_link, html_link on success, or error key on failure.
    """
    from googleapiclient.discovery import build

    creds = get_google_credentials(recruiter_id)
    if not creds:
        return {'error': 'Google Calendar not connected or token expired'}

    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)

    event_body = {
        'summary': summary,
        'description': description,
        'start': {
            'dateTime': start_dt.isoformat(),
            'timeZone': timezone_str,
        },
        'end': {
            'dateTime': end_dt.isoformat(),
            'timeZone': timezone_str,
        },
        'attendees': [{'email': email} for email in attendee_emails],
        'conferenceData': {
            'createRequest': {
                'requestId': str(uuid.uuid4()),
                'conferenceSolutionKey': {'type': 'hangoutsMeet'},
            }
        },
        'reminders': {
            'useDefault': False,
            'overrides': [
                {'method': 'email', 'minutes': 60},
                {'method': 'popup', 'minutes': 15},
            ],
        },
    }

    try:
        service = build('calendar', 'v3', credentials=creds)
        event = service.events().insert(
            calendarId='primary',
            body=event_body,
            conferenceDataVersion=1,
            sendUpdates='all',
        ).execute()

        meet_link = ''
        if event.get('conferenceData', {}).get('entryPoints'):
            for ep in event['conferenceData']['entryPoints']:
                if ep.get('entryPointType') == 'video':
                    meet_link = ep.get('uri', '')
                    break

        return {
            'event_id': event.get('id', ''),
            'meet_link': meet_link,
            'html_link': event.get('htmlLink', ''),
        }
    except Exception as e:
        logger.error(f"Failed to create calendar event: {e}")
        return {'error': str(e)}


def delete_calendar_event(recruiter_id, event_id):
    """Delete a Google Calendar event. Returns dict with success or error."""
    from googleapiclient.discovery import build

    creds = get_google_credentials(recruiter_id)
    if not creds:
        return {'error': 'Google Calendar not connected'}
    try:
        service = build('calendar', 'v3', credentials=creds)
        service.events().delete(
            calendarId='primary', eventId=event_id, sendUpdates='all',
        ).execute()
        return {'success': True}
    except Exception as e:
        logger.error(f"Failed to delete calendar event {event_id}: {e}")
        return {'error': str(e)}
