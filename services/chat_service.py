import os
import logging
from datetime import datetime, timezone

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class CandidateInsightChat:
    """LangChain-based chat interface for candidate-JD Q&A."""

    def __init__(self, chats_collection):
        self.chats_collection = chats_collection
        self.max_history = 10  # Keep last 10 messages for context window

    def _build_system_message(self, insight: dict) -> str:
        """Build compact system message from diff data (~150 tokens)."""
        sd = insight.get('skills_diff', {})
        ed = insight.get('experience_diff', {})
        eud = insight.get('education_diff', {})

        matched = [m['jd_skill'] for m in sd.get('matched', [])]
        missing = sd.get('missing_mandatory', []) + sd.get('missing_optional', [])

        lines = [
            f"You are a senior recruitment analyst. Answer questions about this candidate-job match.",
            f"JOB: {insight.get('job_title', 'N/A')}",
            f"CANDIDATE: {insight.get('candidate_name', 'N/A')}",
            f"SKILLS: {len(matched)} matched, {len(missing)} missing",
            f"MATCHED SKILLS: {', '.join(matched[:10])}",
            f"MISSING SKILLS: {', '.join(missing[:8])}",
            f"EXTRA SKILLS: {', '.join(sd.get('extra', [])[:5])}",
            f"RELEVANT EXP: {ed.get('total_relevant_years', 0)}yr / {ed.get('total_years', 0)}yr total",
            f"EDU: {eud.get('explanation', 'N/A')}",
        ]

        # Add role summaries
        for role in ed.get('roles', [])[:3]:
            lines.append(
                f"ROLE: {role['job_title']} @ {role['company']} "
                f"({role['years']}yr, relevance: {role['relevance_score']})"
            )

        # Add LLM insights if available
        insights = insight.get('llm_insights')
        if insights:
            if insights.get('hiring_recommendation'):
                lines.append(f"RECOMMENDATION: {insights['hiring_recommendation']}")
            if insights.get('strengths'):
                lines.append(f"STRENGTHS: {'; '.join(insights['strengths'][:3])}")

        lines.append("Answer concisely. Base answers on the data above.")
        return "\n".join(lines)

    def _get_llm(self):
        return ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model="llama-3.3-70b-versatile",
            temperature=0.4
        )

    def _get_or_create_session(self, job_id: str, resume_id: str, recruiter_id: str) -> dict:
        session = self.chats_collection.find_one({
            'job_id': job_id,
            'resume_id': resume_id,
            'recruiter_id': recruiter_id
        })
        if not session:
            doc = {
                'job_id': job_id,
                'resume_id': resume_id,
                'recruiter_id': recruiter_id,
                'messages': [],
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
            self.chats_collection.insert_one(doc)
            return doc
        return session

    def send_message(self, job_id: str, resume_id: str, recruiter_id: str,
                     user_message: str, insight: dict) -> dict:
        """Process user question, store in MongoDB, return response."""
        session = self._get_or_create_session(job_id, resume_id, recruiter_id)
        system_msg = self._build_system_message(insight)

        # Build message list for LLM
        messages = [SystemMessage(content=system_msg)]

        # Add history (trimmed to last N messages)
        history = session.get('messages', [])[-self.max_history:]
        for msg in history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))

        messages.append(HumanMessage(content=user_message))

        try:
            llm = self._get_llm()
            response = llm.invoke(messages)
            ai_content = response.content
            tokens = 0
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                tokens = usage.get('total_tokens', 0)

            # Store both messages
            now = datetime.now(timezone.utc)
            new_messages = [
                {'role': 'user', 'content': user_message, 'timestamp': now, 'tokens_used': 0},
                {'role': 'assistant', 'content': ai_content, 'timestamp': now, 'tokens_used': tokens}
            ]

            self.chats_collection.update_one(
                {'job_id': job_id, 'resume_id': resume_id, 'recruiter_id': recruiter_id},
                {
                    '$push': {'messages': {'$each': new_messages}},
                    '$set': {'updated_at': now}
                }
            )

            return {'response': ai_content, 'tokens_used': tokens}

        except Exception as e:
            logger.error(f"Chat error for {resume_id}: {e}")
            raise

    def get_chat_history(self, job_id: str, resume_id: str, recruiter_id: str) -> list:
        """Get chat history for a session."""
        session = self.chats_collection.find_one({
            'job_id': job_id,
            'resume_id': resume_id,
            'recruiter_id': recruiter_id
        })
        if not session:
            return []

        messages = []
        for msg in session.get('messages', []):
            messages.append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat() if msg.get('timestamp') else None
            })
        return messages

    def clear_chat(self, job_id: str, resume_id: str, recruiter_id: str) -> bool:
        """Clear chat session."""
        result = self.chats_collection.update_one(
            {'job_id': job_id, 'resume_id': resume_id, 'recruiter_id': recruiter_id},
            {'$set': {'messages': [], 'updated_at': datetime.now(timezone.utc)}}
        )
        return result.modified_count > 0
