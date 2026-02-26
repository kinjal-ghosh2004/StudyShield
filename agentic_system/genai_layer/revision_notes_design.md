# Structured Revision Notes Module Design

## 1. Overview
The Structured Revision Notes Module operates within the Generative AI Layer to provide targeted academic support immediately upon identifying cognitive bottlenecks. Instead of a general motivational nudge, it constructs a dense, personalized study guide aligned specifically with the student's identified weak topic and the syllabus.

## 2. Trigger Logic
The ReAct Planner agent triggers the Revision Notes Pipeline strictly when:
1. `risk_score > 0.65` (High Risk Zone)
2. `topic_weakness_detected == True` (The Risk Prediction Layer isolated a specific feature or concept, such as "Conceptual Confusion" mapped to a specific module).

**Pseudocode:**
```python
if risk_score > 0.65 and "conceptual_confusion" in root_cause and top_contributing_features:
    target_topic = top_contributing_features[0]
    payload["revision_notes"] = generator.generate_revision_notes(target_topic, syllabus_context)
```

## 3. LLM Prompt Template
To ensure the generative output remains structured and highly actionable, we employ the following system prompt for the LLM call:

```text
You are an expert AI academic tutor. A student is struggling with the topic: {topic_name}.
Based on the {course_syllabus_context}, generate a concise, structured revision note to help them recover.
Your output MUST adhere strictly to the following structure formatted in JSON:
{
  "title": "Revision Note: {topic_name}",
  "key_concepts_summary": ["Concept 1", "Concept 2", "Concept 3 (Limit 2 sentences each)"],
  "simplified_explanation": "A high-level analogy or plain-English breakdown of why this topic matters and how it works.",
  "step_by_step_examples": [
      "Step 1: Action A",
      "Step 2: Action B",
      "Result: Final outcome"
  ],
  "quick_revision_checklist": ["Did I understand X?", "Can I solve Y?", "Have I reviewed Z?"]
}
Keep the tone empathetic but highly analytical. Do not stray from the JSON structure.
```

## 4. Note Generation Pipeline
1. **Context Extraction**: The agent identifies the weak module (e.g., "Module 4.2: Dynamic Programming").
2. **Syllabus Lookup**: The system pulls the relevant learning objectives from a vectorized Syllabus DB.
3. **Prompt Compilation**: The topic and syllabus context drop into the Prompt Template.
4. **LLM Inference**: The GenAI model generates the JSON payload.
5. **UI Rendering**: The JSON is parsed and displayed inside the "Intervention Panel" as a downloadable/interactive "Micro-Study Guide" card.

## 5. Example Output Note

```json
{
  "title": "Revision Note: Dynamic Programming - Knapsack Problem",
  "key_concepts_summary": [
    "Overlapping Subproblems: Breaking down the knapsack into smaller sub-capacities.",
    "Optimal Substructure: The best solution for capacity W relies on the best solution for W-1.",
    "Memoization: Storing the results of these smaller capacities in a 2D array to avoid calculating them twice."
  ],
  "simplified_explanation": "Imagine you are packing for a flight with a strict 15kg baggage limit, and you have 10 items of different weights and values. Instead of guessing every combination, you calculate the best value for a 1kg limit, then 2kg, building up to 15kg using your previous answers to shortcut the math.",
  "step_by_step_examples": [
    "Step 1: Create a 2D array dp[n+1][W+1] initialized to 0.",
    "Step 2: Loop through each item (i) from 1 to n.",
    "Step 3: Loop through each capacity limit (w) from 1 to W.",
    "Step 4: If item weight <= current capacity w, choose max(include_item, exclude_item). Else, exclude_item.",
    "Result: dp[n][W] holds your maximum possible value."
  ],
  "quick_revision_checklist": [
    "Can I write the DP state transition formula from memory?",
    "Do I understand why the array dimensions are (n+1) and (W+1)?",
    "Have I traced a 3-item knapsack problem on paper?"
  ]
}
```
