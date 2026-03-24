"""Hiring pipeline stage helpers."""

DEFAULT_STAGES = ['submitted', 'reviewed', 'shortlisted', 'hired', 'rejected']
FIXED_PREFIX = ['submitted', 'reviewed', 'shortlisted']
FIXED_SUFFIX = ['hired', 'rejected']


def get_pipeline_stages(job):
    """Return pipeline stages for a job, falling back to defaults."""
    return job.get('pipeline_stages') or DEFAULT_STAGES


def validate_stage_transition(current_stage, new_stage, stages):
    """Check that *new_stage* exists in *stages* list.

    We intentionally allow arbitrary transitions (recruiters may skip stages
    or move candidates backward). The only rule is the target must be a valid
    stage for this job.
    """
    return new_stage in stages


def get_notification_stages(job):
    """Return stages that trigger email notifications for a job."""
    return job.get('notification_stages') or [
        s for s in get_pipeline_stages(job) if s != 'submitted'
    ]


def build_pipeline_stages(custom_middle_stages=None):
    """Build a full pipeline list from optional custom middle stages.

    Returns e.g. ['submitted', 'reviewed', 'shortlisted', 'technical_interview', 'hired', 'rejected']
    """
    middle = list(custom_middle_stages) if custom_middle_stages else []
    return FIXED_PREFIX + middle + FIXED_SUFFIX
