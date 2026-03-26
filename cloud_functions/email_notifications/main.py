"""Google Cloud Function: Email Notifications.

HTTP-triggered, 2nd gen. Receives {type, application_id, ...},
looks up data in MongoDB, sends email via Gmail SMTP.
"""

import os
import smtplib
import logging
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import timezone

import functions_framework
from pymongo import MongoClient
from bson.objectid import ObjectId
from dateutil import parser as dateutil_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Module-level globals ─────────────────────────────────────────────
_mongo_client = None
_db = None


def _get_db():
    global _mongo_client, _db
    if _db is None:
        _mongo_client = MongoClient(
            os.environ['MONGO_URI'],
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )
        _db = _mongo_client['resume_parser']
    return _db


def _validate_auth(request):
    expected = os.environ.get('CF_AUTH_TOKEN', '')
    if not expected:
        return False
    auth_header = request.headers.get('Authorization', '')
    return auth_header == f'Bearer {expected}'


def _send_email(recipient, subject, html_body):
    """Send email via Gmail SMTP."""
    mail_user = os.environ.get('MAIL_USERNAME', '')
    mail_pass = os.environ.get('MAIL_PASSWORD', '')
    if not mail_user or not mail_pass:
        logger.warning("Mail credentials not configured — skipping")
        return

    msg = MIMEMultipart('alternative')
    msg['From'] = mail_user
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(mail_user, mail_pass)
        server.sendmail(mail_user, [recipient], msg.as_string())

    logger.info(f"Email sent to {recipient}: {subject}")


def _lookup_app_context(db, application_id):
    """Look up application, user, and job from MongoDB. Returns tuple or None."""
    app_doc = db['application_collections'].find_one({'_id': ObjectId(application_id)})
    if not app_doc:
        logger.warning(f"Application {application_id} not found")
        return None

    user = db['users'].find_one({'_id': app_doc['user_id']})
    if not user or not user.get('email'):
        logger.warning(f"User not found for application {application_id}")
        return None

    job = db['jobs'].find_one({'_id': app_doc['job_id']})
    if not job:
        logger.warning(f"Job not found for application {application_id}")
        return None

    return {
        'applicant_name': user.get('name', user['email'].split('@')[0]),
        'email': user['email'],
        'job_title': job.get('parsed_data', {}).get('title', 'Unknown Position'),
        'company': job.get('company', 'Unknown Company'),
    }


# ── Email Config ─────────────────────────────────────────────────────

STATUS_CONFIG = {
    'reviewed': {
        'subject': 'Your application has been reviewed',
        'heading': 'Application Reviewed',
        'color': '#3B82F6',
        'icon': '&#128065;',
    },
    'shortlisted': {
        'subject': "Great news! You've been shortlisted",
        'heading': "You've Been Shortlisted!",
        'color': '#10B981',
        'icon': '&#11088;',
    },
    'rejected': {
        'subject': 'Update on your application',
        'heading': 'Application Update',
        'color': '#EF4444',
        'icon': '&#128232;',
    },
    'hired': {
        'subject': "Congratulations! You've been hired",
        'heading': 'Congratulations!',
        'color': '#8B5CF6',
        'icon': '&#127881;',
    },
}

STATUS_BODY = {
    'reviewed': 'Your application has been reviewed by our hiring team. We will be in touch with next steps soon.',
    'shortlisted': 'We are excited to let you know that you have been shortlisted for the next stage of our hiring process. Expect to hear from us shortly with further details.',
    'rejected': 'After careful consideration, we have decided to move forward with other candidates at this time. We appreciate your interest and encourage you to apply for future openings.',
    'hired': 'We are thrilled to offer you the position! Our team will reach out with onboarding details very soon.',
}


def _get_status_config(status):
    return STATUS_CONFIG.get(status, {
        'subject': f'Application Update: {status.replace("_", " ").title()}',
        'heading': status.replace('_', ' ').title(),
        'color': '#6B7280',
        'icon': '&#128203;',
    })


def _get_status_body(status):
    return STATUS_BODY.get(status,
        f'Your application status has been updated to: {status.replace("_", " ").title()}. '
        'We will be in touch with more details soon.'
    )


# ── HTML Builders ────────────────────────────────────────────────────

def _build_status_email_html(applicant_name, job_title, company, status, feedback=None):
    cfg = _get_status_config(status)
    body_text = _get_status_body(status)

    feedback_block = ''
    if feedback:
        feedback_block = f'''
            <tr>
                <td style="padding: 20px 30px 0 30px;">
                    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #F9FAFB; border-radius: 8px;">
                        <tr>
                            <td style="padding: 16px 20px;">
                                <p style="margin: 0 0 8px 0; font-size: 13px; font-weight: 600; color: #374151;">Recruiter Feedback</p>
                                <p style="margin: 0; font-size: 14px; color: #4B5563; line-height: 1.5;">{feedback}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>'''

    return f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin: 0; padding: 0; background-color: #F3F4F6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #F3F4F6; padding: 40px 0;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background-color: #FFFFFF; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.07);">
                    <tr>
                        <td style="background-color: {cfg['color']}; padding: 30px; text-align: center;">
                            <div style="font-size: 36px; margin-bottom: 10px;">{cfg['icon']}</div>
                            <h1 style="margin: 0; color: #FFFFFF; font-size: 22px; font-weight: 700;">{cfg['heading']}</h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 30px;">
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #111827;">Hi {applicant_name},</p>
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #374151; line-height: 1.6;">{body_text}</p>
                            <p style="margin: 0; font-size: 14px; color: #6B7280;"><strong>Position:</strong> {job_title}<br><strong>Company:</strong> {company}</p>
                        </td>
                    </tr>
                    {feedback_block}
                    <tr>
                        <td style="padding: 24px 30px; border-top: 1px solid #E5E7EB; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #9CA3AF;">This is an automated notification from Emploify.io</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>'''


def _build_interview_email_html(applicant_name, job_title, company, interview_datetime, duration_minutes, meet_link=None, notes=None):
    date_str = interview_datetime.strftime('%A, %B %d, %Y')
    time_str = interview_datetime.strftime('%I:%M %p')

    meet_block = ''
    if meet_link:
        meet_block = f'''
            <tr>
                <td style="padding: 20px 30px 0 30px; text-align: center;">
                    <a href="{meet_link}" style="display: inline-block; padding: 12px 28px; background-color: #1a73e8; color: #FFFFFF; text-decoration: none; border-radius: 6px; font-size: 15px; font-weight: 600;">
                        Join Google Meet
                    </a>
                    <p style="margin: 10px 0 0 0; font-size: 12px; color: #6B7280;">{meet_link}</p>
                </td>
            </tr>'''

    notes_block = ''
    if notes:
        notes_block = f'''
            <tr>
                <td style="padding: 20px 30px 0 30px;">
                    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #F9FAFB; border-radius: 8px;">
                        <tr>
                            <td style="padding: 16px 20px;">
                                <p style="margin: 0 0 8px 0; font-size: 13px; font-weight: 600; color: #374151;">Additional Notes</p>
                                <p style="margin: 0; font-size: 14px; color: #4B5563; line-height: 1.5;">{notes}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>'''

    return f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin: 0; padding: 0; background-color: #F3F4F6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #F3F4F6; padding: 40px 0;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background-color: #FFFFFF; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.07);">
                    <tr>
                        <td style="background-color: #1a73e8; padding: 30px; text-align: center;">
                            <div style="font-size: 36px; margin-bottom: 10px;">&#128197;</div>
                            <h1 style="margin: 0; color: #FFFFFF; font-size: 22px; font-weight: 700;">Interview Scheduled</h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 30px;">
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #111827;">Hi {applicant_name},</p>
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #374151; line-height: 1.6;">We are pleased to invite you for an interview for the <strong>{job_title}</strong> position at <strong>{company}</strong>.</p>
                            <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #EFF6FF; border-radius: 8px; margin-top: 10px;">
                                <tr>
                                    <td style="padding: 20px;">
                                        <p style="margin: 0 0 8px 0; font-size: 14px; color: #374151;"><strong>Date:</strong> {date_str}</p>
                                        <p style="margin: 0 0 8px 0; font-size: 14px; color: #374151;"><strong>Time:</strong> {time_str}</p>
                                        <p style="margin: 0; font-size: 14px; color: #374151;"><strong>Duration:</strong> {duration_minutes} minutes</p>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    {meet_block}
                    {notes_block}
                    <tr>
                        <td style="padding: 24px 30px; border-top: 1px solid #E5E7EB; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #9CA3AF;">This is an automated notification from Emploify.io</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>'''


def _build_cancellation_email_html(applicant_name, job_title, company, interview_datetime):
    date_str = interview_datetime.strftime('%A, %B %d, %Y')
    time_str = interview_datetime.strftime('%I:%M %p')

    return f'''<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin: 0; padding: 0; background-color: #F3F4F6; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #F3F4F6; padding: 40px 0;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background-color: #FFFFFF; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.07);">
                    <tr>
                        <td style="background-color: #EF4444; padding: 30px; text-align: center;">
                            <div style="font-size: 36px; margin-bottom: 10px;">&#10060;</div>
                            <h1 style="margin: 0; color: #FFFFFF; font-size: 22px; font-weight: 700;">Interview Cancelled</h1>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 30px;">
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #111827;">Hi {applicant_name},</p>
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #374151; line-height: 1.6;">We regret to inform you that the scheduled interview for the <strong>{job_title}</strong> position at <strong>{company}</strong> has been cancelled.</p>
                            <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #FEF2F2; border-radius: 8px; margin-top: 10px;">
                                <tr>
                                    <td style="padding: 20px;">
                                        <p style="margin: 0 0 8px 0; font-size: 14px; color: #374151;"><strong>Original Date:</strong> {date_str}</p>
                                        <p style="margin: 0; font-size: 14px; color: #374151;"><strong>Original Time:</strong> {time_str}</p>
                                    </td>
                                </tr>
                            </table>
                            <p style="margin: 16px 0 0 0; font-size: 14px; color: #6B7280; line-height: 1.5;">The recruiter may reach out to reschedule. Please check your application status for updates.</p>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 24px 30px; border-top: 1px solid #E5E7EB; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #9CA3AF;">This is an automated notification from Emploify.io</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>'''


# ── Handlers by type ─────────────────────────────────────────────────

def _handle_status(db, data):
    application_id = data.get('application_id')
    status = data.get('status')
    feedback = data.get('feedback')

    if not application_id or not status:
        return ({'error': 'application_id and status are required'}, 400)

    ctx = _lookup_app_context(db, application_id)
    if not ctx:
        return ({'error': 'Application/user/job not found'}, 404)

    cfg = _get_status_config(status)
    html = _build_status_email_html(ctx['applicant_name'], ctx['job_title'], ctx['company'], status, feedback)
    _send_email(ctx['email'], f"{cfg['subject']} — {ctx['job_title']}", html)
    return ({'status': 'sent'}, 200)


def _handle_interview(db, data):
    application_id = data.get('application_id')
    interview_datetime_str = data.get('interview_datetime')
    duration_minutes = data.get('duration_minutes', 30)
    meet_link = data.get('meet_link')
    notes = data.get('notes')

    if not application_id or not interview_datetime_str:
        return ({'error': 'application_id and interview_datetime are required'}, 400)

    ctx = _lookup_app_context(db, application_id)
    if not ctx:
        return ({'error': 'Application/user/job not found'}, 404)

    interview_datetime = dateutil_parser.parse(interview_datetime_str)
    html = _build_interview_email_html(
        ctx['applicant_name'], ctx['job_title'], ctx['company'],
        interview_datetime, duration_minutes, meet_link, notes,
    )
    _send_email(ctx['email'], f"Interview Scheduled — {ctx['job_title']} at {ctx['company']}", html)
    return ({'status': 'sent'}, 200)


def _handle_cancellation(db, data):
    application_id = data.get('application_id')
    interview_datetime_str = data.get('interview_datetime')

    if not application_id or not interview_datetime_str:
        return ({'error': 'application_id and interview_datetime are required'}, 400)

    ctx = _lookup_app_context(db, application_id)
    if not ctx:
        return ({'error': 'Application/user/job not found'}, 404)

    interview_datetime = dateutil_parser.parse(interview_datetime_str)
    html = _build_cancellation_email_html(
        ctx['applicant_name'], ctx['job_title'], ctx['company'], interview_datetime,
    )
    _send_email(ctx['email'], f"Interview Cancelled — {ctx['job_title']} at {ctx['company']}", html)
    return ({'status': 'sent'}, 200)


_HANDLERS = {
    'status': _handle_status,
    'interview': _handle_interview,
    'cancellation': _handle_cancellation,
}


# ── Entry point ──────────────────────────────────────────────────────

@functions_framework.http
def send_notification(request):
    """GCF entry point — HTTP POST with JSON body."""
    if request.method == 'OPTIONS':
        return ('', 204, {'Access-Control-Allow-Origin': '*',
                          'Access-Control-Allow-Methods': 'POST',
                          'Access-Control-Allow-Headers': 'Authorization, Content-Type'})

    if request.method != 'POST':
        return ({'error': 'Method not allowed'}, 405)

    if not _validate_auth(request):
        return ({'error': 'Unauthorized'}, 401)

    data = request.get_json(silent=True)
    if not data:
        return ({'error': 'Missing JSON body'}, 400)

    email_type = data.get('type')
    handler = _HANDLERS.get(email_type)
    if not handler:
        return ({'error': f'Unknown type: {email_type}. Must be one of: {list(_HANDLERS.keys())}'}, 400)

    db = _get_db()

    try:
        return handler(db, data)
    except Exception as e:
        logger.error(f"Email notification failed ({email_type}): {e}")
        return ({'error': str(e)}, 500)
