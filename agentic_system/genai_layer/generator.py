class InterventionGenerator:
    """
    Generative AI Layer responsible for formatting personalized messages to the student
    based on the Agentic Layer's selected strategy and diagnosed cause.
    """
    def generate(self, strategy: str, root_cause: str, top_features: list) -> dict:
        """
        Simulates an LLM structured generation call.
        """
        print(f"\n[GenAI] Calling LLM API for strategy: '{strategy}'...")
        
        explanation = ""
        study_plan = ""
        motivation = ""
        remedial_task = ""

        if strategy == "Conceptual Breakdown":
            explanation = "It looks like you might be stuck on recent physics formulas. Let's break down Newton's Second Law into simpler terms."
            remedial_task = "Review interactive module 4.1: Force and Mass."
            motivation = "Many students get stuck here. A quick review is all it takes to get back on track!"
            
        elif strategy == "Micro-Plan":
            study_plan = "Here is a 3-day micro schedule:\nDay 1: Read Chapter 4 (20 mins)\nDay 2: Complete Quiz 4a (15 mins)\nDay 3: Attempt Assignment (25 mins)"
            motivation = "Breaking things down makes them manageable. You've got this!"
            remedial_task = "Set a calendar reminder for 20 minutes tonight."

        elif strategy == "Motivation Boost":
            motivation = "We noticed you haven't been as active lately. Remember your goal of mastering Data Structures! Just logging in today is a great step."
            remedial_task = "Log into the forum and answer one peer question."
            
        elif strategy == "Human Evaluation":
            explanation = "Your learning profile has been flagged for human review to provide you with the best possible support."
            motivation = "An academic advisor will reach out shortly to help you map out your success path."
            remedial_task = "Check your student email for a meeting link."
            
        else:
            explanation = f"We noticed {root_cause}."
            motivation = "We're here to support your learning journey."

        return {
            "explanation": explanation,
            "study_plan": study_plan,
            "motivation": motivation,
            "remedial_task": remedial_task
        }

    def generate_revision_notes(self, topic_name: str) -> dict:
        """
        Dynamically generates a structured revision note tailored to a specific weak topic.
        """
        print(f"  [GenAI] Generating Structured Revision Notes for topic: '{topic_name}'...")
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

    def generate_adaptive_schedule(self, weak_topic: str) -> dict:
        """
        Dynamically restructures a student's weekly schedule to relieve cognitive load.
        """
        print(f"  [GenAI] Generating Adaptive Schedule tailored to '{weak_topic}'...")
        return {
            "title": "Adaptive Restructuring: Adjusted Weekly Schedule",
            "rationale": "We noticed you're putting in the time but hitting some friction. We've temporarily reduced your daily reading load and shifted advanced topics to next week so you can master the core concepts without feeling rushed.",
            "schedule": [
                {
                    "day": "Today",
                    "focus": "Core Revision",
                    "tasks": [
                        {"time_estimate": "20 mins", "action": f"Review interactive module on {weak_topic} (Weakness Detected)"},
                        {"time_estimate": "30 mins", "action": "Complete basic practice set"}
                    ]
                },
                {
                    "day": "Tomorrow",
                    "focus": "New Material (Reduced Load)",
                    "tasks": [
                        {"time_estimate": "45 mins", "action": "Watch core lecture"},
                        {"time_estimate": "15 mins", "action": "Read chapter summary (Skipping deep-dive proofs for now)"}
                    ]
                },
                {
                    "day": "Day 3",
                    "focus": "Synthesis",
                    "tasks": [
                        {"time_estimate": "30 mins", "action": "Attempt Assignment (Deadline extended by 48h)"}
                    ]
                }
            ],
            "deferred_topics": [
                "Advanced Proofs (Moved to Next Week)",
                "Edge Case Mechanics (Moved to Next Week)"
            ]
        }

if __name__ == "__main__":
    generator = InterventionGenerator()
    payload = generator.generate("Micro-Plan", "Student pace is low", ["low_pace"])
    import json
    print(json.dumps(payload, indent=2))
