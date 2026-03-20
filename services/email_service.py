import logging
from flask import current_app
from flask_mail import Message
from extensions import mail, applications_collection, users_collection, jobs_collection, task_runner
from bson.objectid import ObjectId

logger = logging.getLogger(__name__)

STATUS_CONFIG = {
    'reviewed': {
        'subject': 'Your application has been reviewed',
        'heading': 'Application Reviewed',
        'color': '#3B82F6',
        'icon': '&#128065;',
    },
    'shortlisted': {
        'subject': 'Great news! You\'ve been shortlisted',
        'heading': 'You\'ve Been Shortlisted!',
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
        'subject': 'Congratulations! You\'ve been hired',
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


def build_status_email_html(applicant_name, job_title, company, status, feedback=None):
    cfg = STATUS_CONFIG[status]
    body_text = STATUS_BODY[status]

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
                    <!-- Header -->
                    <tr>
                        <td style="background-color: {cfg['color']}; padding: 30px; text-align: center;">
                            <div style="font-size: 36px; margin-bottom: 10px;">{cfg['icon']}</div>
                            <h1 style="margin: 0; color: #FFFFFF; font-size: 22px; font-weight: 700;">{cfg['heading']}</h1>
                        </td>
                    </tr>
                    <!-- Body -->
                    <tr>
                        <td style="padding: 30px;">
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #111827;">Hi {applicant_name},</p>
                            <p style="margin: 0 0 16px 0; font-size: 15px; color: #374151; line-height: 1.6;">{body_text}</p>
                            <p style="margin: 0; font-size: 14px; color: #6B7280;"><strong>Position:</strong> {job_title}<br><strong>Company:</strong> {company}</p>
                        </td>
                    </tr>
                    {feedback_block}
                    <!-- Footer -->
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


def send_status_notification(app, application_id, status, feedback=None):
    """Runs inside a background thread. Sends email notification for status change."""
    try:
        with app.app_context():
            if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
                logger.info("Mail credentials not configured — skipping notification")
                return

            application = applications_collection.find_one({'_id': ObjectId(application_id)})
            if not application:
                logger.warning(f"Application {application_id} not found for email notification")
                return

            user = users_collection.find_one({'_id': application['user_id']})
            if not user or not user.get('email'):
                logger.warning(f"User not found or no email for application {application_id}")
                return

            job = jobs_collection.find_one({'_id': application['job_id']})
            if not job:
                logger.warning(f"Job not found for application {application_id}")
                return

            job_title = job.get('parsed_data', {}).get('title', 'Unknown Position')
            company = job.get('company', 'Unknown Company')
            applicant_name = user.get('name', user['email'].split('@')[0])

            cfg = STATUS_CONFIG[status]
            html = build_status_email_html(applicant_name, job_title, company, status, feedback)

            msg = Message(
                subject=f"{cfg['subject']} — {job_title}",
                recipients=[user['email']],
                html=html,
            )
            mail.send(msg)
            logger.info(f"Status notification sent to {user['email']} (status={status})")

    except Exception as e:
        logger.error(f"Failed to send status notification for application {application_id}: {e}")


def dispatch_status_email(application_id, status, feedback=None):
    """Non-blocking dispatch — submits email task to the background TaskRunner."""
    if status not in STATUS_CONFIG:
        return

    app = current_app._get_current_object()
    task_runner.submit(send_status_notification, app, application_id, status, feedback)
