# Personalized Learning Pace Optimizer Design

## 1. Overview
The Personalized Learning Pace Optimizer addresses structural burnout and cognitive overload. When the AI detects a student struggling with pace (spending extensive hours but yielding poor performance, accompanied by high session fragmentation), it autonomously restructures their upcoming weekly syllabus.

Instead of a generic "study harder" nudge, it actually acts to reduce the immediate cognitive burden by pushing back deadlines for advanced topics and injecting immediate micro-revisions.

## 2. Detection Criteria (Cognitive Overload Trigger)
The optimizer triggers under the following conditions synthesized from the Risk Predictor and ReAct Agent:
- `root_cause` == "Burnout" OR "Confusion / Cognitive Overload"
- `state.drift_vector.volatility` > 2.0 (High session fragmentation)
- `state.drift_vector.lag` > 3.0 (Significant assignment backlog forming)

## 3. Constraint Modeling & Optimization Logic
The generated schedule must balance the student's available bandwidth against the strict requirements of the course.

**Constraints:**
- $B_{max}$: Maximum daily cognitive bandwidth (e.g., 2 hours/day).
- $D_{hard}$: Absolute hard deadline for the module exam.
- $W_{core}$: Core foundational topics that cannot be skipped.
- $W_{advanced}$: Advanced topics that can be deferred if pacing requires it.

**Action Mapping:**
1. **Reduce Daily Workload**: Strip advanced readings, focusing only on $W_{core}$.
2. **Reschedule Weak Topics**: Inject a 30-minute revision block on yesterday's failed quiz topic before introducing new material today.
3. **Adjust Deadline Spacing**: Push the nearest assignment deadline out by 48 hours (if allowed by $D_{hard}$), spacing out the required reading.

## 4. Adaptive Schedule Generation Pipeline
1. The ReAct Agent selects the `"Micro-Plan"` strategy.
2. The `InterventionGenerator` detects the strategy and triggers `generate_adaptive_schedule()`.
3. The LLM translates the constraints into a JSON-formatted adaptive weekly schedule.
4. The frontend renders this JSON as an interactive calendar in the intervention panel.

## 5. Example UI Output Payload

```json
{
  "title": "Adaptive Restructuring: Week 4 Schedule",
  "rationale": "We noticed you're spending a lot of time on the platform but struggling with the recent quizzes. We've temporarily reduced your daily reading load by 30% and shifted the advanced topics to next week to give you time to master the core concepts.",
  "schedule": [
    {
      "day": "Monday (Today)",
      "focus": "Core Revision",
      "tasks": [
        {"time_estimate": "20 mins", "action": "Review interactive module on Matrix Multiplication (Weakness Detected)"},
        {"time_estimate": "30 mins", "action": "Complete basic practice set 4.1"}
      ]
    },
    {
      "day": "Tuesday",
      "focus": "New Material (Reduced Load)",
      "tasks": [
        {"time_estimate": "45 mins", "action": "Watch lecture: Introduction to Eigenvectors"},
        {"time_estimate": "15 mins", "action": "Read chapter summary (Skipping deep-dive proofs for now)"}
      ]
    },
    {
      "day": "Wednesday",
      "focus": "Synthesis",
      "tasks": [
        {"time_estimate": "30 mins", "action": "Attempt Assignment 4 (Deadline extended by 48h)"}
      ]
    }
  ],
  "deferred_topics": [
    "Eigenvalue Proofs (Moved to Week 5)",
    "Advanced Transformations (Moved to Week 5)"
  ]
}
```
