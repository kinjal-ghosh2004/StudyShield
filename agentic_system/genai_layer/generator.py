"""
GenAI Layer — InterventionGenerator
Uses Google Gemini to produce structured JSON interventions tailored to each student's risk profile.
"""
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _API_KEY = os.getenv("GEMINI_API_KEY", "")
    if _API_KEY and _API_KEY != "your_gemini_api_key_here":
        genai.configure(api_key=_API_KEY)
        _model = genai.GenerativeModel("gemini-1.5-flash")
        _GENAI_AVAILABLE = True
        logger.info("Gemini API configured successfully.")
    else:
        logger.warning("GEMINI_API_KEY not set — falling back to template mode.")
        _GENAI_AVAILABLE = False
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed — falling back to template mode.")


def _call_gemini(prompt: str, model=None) -> dict:
    """Call Gemini and parse a JSON response. Raises on failure."""
    m = model or _model
    response = m.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    text = response.text.strip()
    return json.loads(text)


class InterventionGenerator:
    """
    GenAI Layer responsible for crafting personalized, pedagogically sound interventions.
    Uses Gemini when available; falls back to structured templates.
    """

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(self, strategy: str, root_cause: str, top_features: list) -> dict:
        """Generate a personalised intervention message for the chosen strategy."""
        if _GENAI_AVAILABLE:
            return self._gemini_generate(strategy, root_cause, top_features)
        return self._template_generate(strategy, root_cause)

    def generate_revision_notes(self, topic_name: str) -> dict:
        """Produce structured revision notes for a weak topic."""
        if _GENAI_AVAILABLE:
            return self._gemini_revision_notes(topic_name)
        return self._template_revision_notes(topic_name)

    def generate_adaptive_schedule(self, weak_topic: str) -> dict:
        """Restructure a student's weekly schedule to reduce cognitive load."""
        if _GENAI_AVAILABLE:
            return self._gemini_adaptive_schedule(weak_topic)
        return self._template_adaptive_schedule(weak_topic)

    # ── Gemini Implementations ───────────────────────────────────────────────

    def _gemini_generate(self, strategy: str, root_cause: str, top_features: list) -> dict:
        prompt = f"""
You are an expert educational psychologist AI helping an e-learning platform support at-risk students.

A student has been identified as at risk of dropping out.
- Diagnosed Root Cause: {root_cause}
- Recommended Intervention Strategy: {strategy}
- Top Contributing Behavioral Signals: {", ".join(top_features) if top_features else "general decline"}

Generate a compassionate, personalised intervention message. Return ONLY valid JSON in the following schema:
{{
  "explanation": "A brief, student-friendly explanation of what pattern we observed (1-2 sentences, no jargon).",
  "study_plan": "A concrete 3-day micro study plan matching the strategy. Empty string if not applicable.",
  "motivation": "An empathetic, motivating message tailored to the strategy (1-2 sentences).",
  "remedial_task": "One specific, achievable action the student can take in the next 30 minutes."
}}
"""
        try:
            result = _call_gemini(prompt)
            logger.info(f"Gemini generate() succeeded for strategy={strategy}")
            return result
        except Exception as e:
            logger.error(f"Gemini generate() failed: {e}")
            return self._template_generate(strategy, root_cause)

    def _gemini_revision_notes(self, topic_name: str) -> dict:
        prompt = f"""
You are an expert e-learning content designer. Generate concise structured revision notes for a student struggling with the topic: "{topic_name}".

Return ONLY valid JSON matching this schema:
{{
  "title": "Revision Note: {topic_name}",
  "key_concepts_summary": ["concept 1", "concept 2", "concept 3"],
  "simplified_explanation": "A simple analogy or plain-English explanation (2-3 sentences).",
  "step_by_step_examples": ["Step 1: ...", "Step 2: ...", "Result: ..."],
  "quick_revision_checklist": ["Can I define it in one sentence?", "Can I recognize it in a problem?", "Have I done the practice exercise?"]
}}
"""
        try:
            result = _call_gemini(prompt)
            logger.info(f"Gemini revision_notes() succeeded for topic={topic_name}")
            return result
        except Exception as e:
            logger.error(f"Gemini revision_notes() failed: {e}")
            return self._template_revision_notes(topic_name)

    def _gemini_adaptive_schedule(self, weak_topic: str) -> dict:
        prompt = f"""
You are an e-learning pace optimisation system. A student is struggling with "{weak_topic}".
Create a realistic 3-day adaptive study schedule that reduces their cognitive load while keeping them on track.

Return ONLY valid JSON matching this schema:
{{
  "title": "Adaptive Restructuring: Adjusted Weekly Schedule",
  "rationale": "One sentence explaining why the schedule was adjusted.",
  "schedule": [
    {{"day": "Today", "focus": "...", "tasks": [{{"time_estimate": "X mins", "action": "..."}}]}},
    {{"day": "Tomorrow", "focus": "...", "tasks": [{{"time_estimate": "X mins", "action": "..."}}]}},
    {{"day": "Day 3", "focus": "...", "tasks": [{{"time_estimate": "X mins", "action": "..."}}]}}
  ],
  "deferred_topics": ["Topic A (moved to next week)", "Topic B (moved to next week)"]
}}
"""
        try:
            result = _call_gemini(prompt)
            logger.info(f"Gemini adaptive_schedule() succeeded for topic={weak_topic}")
            return result
        except Exception as e:
            logger.error(f"Gemini adaptive_schedule() failed: {e}")
            return self._template_adaptive_schedule(weak_topic)

    # ── Template Fallbacks ───────────────────────────────────────────────────

    def _template_generate(self, strategy: str, root_cause: str) -> dict:
        templates = {
            "Conceptual Breakdown": {
                "explanation": "It looks like you might be stuck on some recent concepts. Let's break things down into simpler steps.",
                "study_plan": "",
                "motivation": "Many students get stuck here. A quick review is all it takes to get back on track!",
                "remedial_task": "Review the interactive module for the most recent topic."
            },
            "Micro-Plan": {
                "explanation": "We noticed your study sessions have been shorter lately.",
                "study_plan": "Day 1: Read Chapter (20 mins)\nDay 2: Complete Quiz (15 mins)\nDay 3: Attempt Assignment (25 mins)",
                "motivation": "Breaking things down makes them manageable. You've got this!",
                "remedial_task": "Set a calendar reminder for 20 minutes tonight."
            },
            "Motivation Boost": {
                "explanation": "",
                "study_plan": "",
                "motivation": f"We noticed you haven't been as active lately. Remember your goals — just logging in today is a great step!",
                "remedial_task": "Log into the forum and answer one peer question."
            },
            "Human Escalation": {
                "explanation": "Your learning profile has been flagged for human review to provide you with the best possible support.",
                "study_plan": "",
                "motivation": "An academic advisor will reach out shortly to help you map out your success path.",
                "remedial_task": "Check your student email for a meeting link."
            },
        }
        return templates.get(strategy, {
            "explanation": f"We noticed {root_cause}.",
            "study_plan": "",
            "motivation": "We're here to support your learning journey.",
            "remedial_task": "Reach out to your course instructor for guidance."
        })

    def _template_revision_notes(self, topic_name: str) -> dict:
        return {
            "title": f"Revision Note: {topic_name}",
            "key_concepts_summary": [
                f"Core principle of {topic_name}.",
                "Common pitfalls to avoid.",
                "How it connects to the broader course objectives."
            ],
            "simplified_explanation": f"Imagine {topic_name} as a simple daily task. Breaking it down helps understand the mechanics without the complex terminology.",
            "step_by_step_examples": [
                "Step 1: Identify the core variables.",
                "Step 2: Apply the standard formula.",
                "Result: Verify the outcome against common sense bounds."
            ],
            "quick_revision_checklist": [
                "Can I define it in one sentence?",
                "Can I recognize a problem that requires this concept?",
                "Have I successfully completed the practice exercise?"
            ]
        }

    def _template_adaptive_schedule(self, weak_topic: str) -> dict:
        return {
            "title": "Adaptive Restructuring: Adjusted Weekly Schedule",
            "rationale": "Temporarily reduced load to allow mastery of core concepts.",
            "schedule": [
                {"day": "Today", "focus": "Core Revision", "tasks": [
                    {"time_estimate": "20 mins", "action": f"Review interactive module on {weak_topic}"},
                    {"time_estimate": "30 mins", "action": "Complete basic practice set"}
                ]},
                {"day": "Tomorrow", "focus": "New Material (Reduced Load)", "tasks": [
                    {"time_estimate": "45 mins", "action": "Watch core lecture"},
                    {"time_estimate": "15 mins", "action": "Read chapter summary"}
                ]},
                {"day": "Day 3", "focus": "Synthesis", "tasks": [
                    {"time_estimate": "30 mins", "action": "Attempt Assignment"}
                ]}
            ],
            "deferred_topics": ["Advanced Proofs (Moved to Next Week)", "Edge Cases (Moved to Next Week)"]
        }


if __name__ == "__main__":
    gen = InterventionGenerator()
    import json
    result = gen.generate("Micro-Plan", "Student pace has declined significantly", ["low_pace", "lag"])
    print(json.dumps(result, indent=2))
