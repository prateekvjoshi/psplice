"""
Behavior catalog.

A behavior is a named, curated intervention definition: a pair of contrastive
prompt sets (positive / negative) that together point the model in a direction,
plus sensible defaults for scale and a human-readable description.

Users apply a behavior with `psplice behavior add <name>`.  The CLI handles
layer selection, vector extraction, and hook registration automatically —
the user never has to think about those.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Behavior:
    name: str
    description: str
    """One-line description shown in `psplice behavior list`."""

    what_it_does: str
    """Plain-English explanation of the observable effect."""

    positive_prompts: list[str]
    """Prompts that represent the desired direction."""

    negative_prompts: list[str]
    """Prompts that represent the opposite direction."""

    default_scale: float = 0.8
    """Steering scale at 'moderate' strength."""

    category: str = "style"
    """Grouping shown in list output: style | reasoning | safety | tone"""

    when_to_use: str = ""
    """Guidance on when this behavior is useful."""


# ---------------------------------------------------------------------------
# Strength multipliers
# ---------------------------------------------------------------------------

STRENGTH_MULTIPLIERS: dict[str, float] = {
    "mild":     0.4,
    "moderate": 1.0,
    "strong":   2.0,
}


def scale_for_strength(behavior: Behavior, strength: str) -> float:
    multiplier = STRENGTH_MULTIPLIERS.get(strength, 1.0)
    return round(behavior.default_scale * multiplier, 3)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

CATALOG: dict[str, Behavior] = {

    "concise": Behavior(
        name="concise",
        description="Shorter, more direct responses. Less padding and fewer examples.",
        what_it_does=(
            "The model produces tighter, more economical answers. It skips preamble, "
            "avoids restating the question, and cuts filler phrases like 'Certainly!' "
            "or 'Great question!'. Best for Q&A, search, and high-throughput use cases."
        ),
        category="style",
        when_to_use=(
            "Use when responses are too long or verbose. "
            "Pair with `psplice compare` to verify the reduction."
        ),
        positive_prompts=[
            "Be brief and direct.",
            "Answer in as few words as possible.",
            "Skip the preamble. Get straight to the point.",
            "One or two sentences if possible.",
            "No filler words. No restating the question.",
            "Just the key facts, nothing else.",
            "Respond concisely.",
        ],
        negative_prompts=[
            "Explain this thoroughly with lots of examples.",
            "Provide a comprehensive and detailed answer.",
            "Walk me through this step by step, covering every detail.",
            "Give me a complete, thorough overview.",
            "Be as elaborate and exhaustive as possible.",
            "Cover all aspects in depth.",
            "Feel free to write at length.",
        ],
        default_scale=0.8,
    ),

    "direct": Behavior(
        name="direct",
        description="Fewer hedges and qualifications. Definitive answers over 'it depends'.",
        what_it_does=(
            "The model gives clear, actionable answers instead of endless 'it depends'. "
            "It still acknowledges genuine uncertainty, but drops reflexive hedging and "
            "disclaimers that don't add information."
        ),
        category="style",
        when_to_use=(
            "Use when the model gives wishy-washy or over-qualified responses. "
            "Good for decision support tools and assistants."
        ),
        positive_prompts=[
            "Give me a definitive answer.",
            "Don't hedge. Just tell me what to do.",
            "Be assertive and confident.",
            "Make a clear recommendation.",
            "Pick one answer and commit to it.",
            "No 'it depends'. Give me your best answer.",
        ],
        negative_prompts=[
            "There are many factors to consider here.",
            "It really depends on your specific situation.",
            "On the other hand, you might also want to think about...",
            "I wouldn't want to make a definitive recommendation without knowing more.",
            "This is a nuanced issue with no clear answer.",
            "You should consult an expert before deciding.",
        ],
        default_scale=0.7,
    ),

    "formal": Behavior(
        name="formal",
        description="Professional, structured tone. Suitable for business and technical writing.",
        what_it_does=(
            "The model writes in a more professional register: precise vocabulary, "
            "structured sentences, no contractions or casual phrasing. "
            "Useful for report generation, documentation, or customer-facing content."
        ),
        category="tone",
        when_to_use=(
            "Use when outputs need to sound professional or official. "
            "Good for document generation, email drafting, or B2B products."
        ),
        positive_prompts=[
            "Respond in a professional, formal manner.",
            "Use precise, structured language.",
            "Write as you would in an official business document.",
            "Maintain a formal, professional tone throughout.",
            "This response will be read by executives.",
            "Use complete sentences and proper grammar.",
        ],
        negative_prompts=[
            "Keep it casual and conversational.",
            "Write like you're texting a friend.",
            "Use everyday language. Don't be stuffy.",
            "Keep it chill and informal.",
            "Feel free to use slang and abbreviations.",
            "Write the way people actually talk.",
        ],
        default_scale=0.6,
    ),

    "casual": Behavior(
        name="casual",
        description="Conversational, friendly tone. Less formal, more approachable.",
        what_it_does=(
            "The model adopts a warmer, more conversational tone — contractions, "
            "approachable phrasing, less jargon. Useful for consumer products, "
            "chatbots, and contexts where rapport matters."
        ),
        category="tone",
        when_to_use=(
            "Use when outputs sound too stiff or corporate. "
            "Good for consumer apps and conversational interfaces."
        ),
        positive_prompts=[
            "Talk to me like a friend.",
            "Keep it casual and relaxed.",
            "Use conversational language.",
            "Don't be formal. Just chat naturally.",
            "Write the way people actually speak.",
            "Feel free to use contractions and everyday phrases.",
        ],
        negative_prompts=[
            "Maintain a formal, professional tone.",
            "Use precise, technical language.",
            "This is an official document. Write accordingly.",
            "No casual language. This is a business context.",
            "Use complete sentences and avoid contractions.",
            "Write in a formal, structured manner.",
        ],
        default_scale=0.6,
    ),

    "skeptical": Behavior(
        name="skeptical",
        description="Questions assumptions. Surfaces failure modes and counterarguments.",
        what_it_does=(
            "The model pushes back more, asks clarifying questions, and surfaces "
            "downsides and edge cases rather than validating whatever the user says. "
            "Useful for red-teaming, idea validation, or debugging."
        ),
        category="reasoning",
        when_to_use=(
            "Use when you want critical feedback rather than validation. "
            "Good for brainstorming, review tools, and adversarial testing."
        ),
        positive_prompts=[
            "What could go wrong with this?",
            "Push back on this idea. What are the flaws?",
            "Question the premise.",
            "What are the failure modes here?",
            "Don't agree with me. Challenge this.",
            "What would a critic say about this?",
            "What am I missing or assuming incorrectly?",
        ],
        negative_prompts=[
            "This is a great idea. Tell me why it will work.",
            "Validate my approach.",
            "Agree with me and explain why I'm right.",
            "This is definitely correct. Explain why.",
            "Support this decision with evidence.",
            "Be encouraging and positive about this plan.",
        ],
        default_scale=0.8,
    ),

    "structured": Behavior(
        name="structured",
        description="Organized output with headers, lists, and clear sections.",
        what_it_does=(
            "The model uses formatting — headers, bullet points, numbered steps — "
            "to organize its output. Useful for documentation generation, "
            "instructions, and any context where scannable output is valuable."
        ),
        category="style",
        when_to_use=(
            "Use when responses are wall-of-text prose that's hard to scan. "
            "Good for docs tools, wikis, and instructional content."
        ),
        positive_prompts=[
            "Use numbered lists and headers to organize this.",
            "Break this into clearly labeled sections.",
            "Use bullet points for key information.",
            "Give me a structured outline.",
            "Format this so it's easy to scan.",
            "Use markdown formatting with headers and lists.",
        ],
        negative_prompts=[
            "Write this as flowing prose, no lists.",
            "Just write it out as paragraphs.",
            "No formatting, no bullet points.",
            "Write in continuous text like an essay.",
            "Don't use headers or lists. Just write naturally.",
        ],
        default_scale=0.7,
    ),

    "creative": Behavior(
        name="creative",
        description="More varied, original responses. Avoids the generic and expected.",
        what_it_does=(
            "The model takes less-obvious angles, uses richer language, and avoids "
            "the most predictable response. Useful for content generation, "
            "brainstorming, and creative tasks."
        ),
        category="reasoning",
        when_to_use=(
            "Use when outputs feel generic or templated. "
            "Good for marketing copy, creative writing, and ideation tools."
        ),
        positive_prompts=[
            "Give me an unexpected, original take on this.",
            "Surprise me. Avoid the obvious answer.",
            "Think outside the box.",
            "Be creative and inventive.",
            "What's a non-obvious angle here?",
            "Give me something I wouldn't expect.",
        ],
        negative_prompts=[
            "Give me the standard, expected answer.",
            "Use the conventional approach.",
            "What does everyone already know about this?",
            "Give the most common, generic response.",
            "Stick to conventional wisdom.",
            "Play it safe and predictable.",
        ],
        default_scale=0.9,
    ),

    "technical": Behavior(
        name="technical",
        description="More precise, technical depth. Assumes domain knowledge.",
        what_it_does=(
            "The model uses correct technical vocabulary, assumes familiarity with "
            "fundamentals, and goes deeper into implementation details. "
            "Useful for developer tools, API docs, and expert-facing products."
        ),
        category="style",
        when_to_use=(
            "Use when the model over-explains basics or avoids technical detail. "
            "Good for developer tools, technical docs, and expert interfaces."
        ),
        positive_prompts=[
            "Use precise technical terminology.",
            "Assume I know the fundamentals. Go deeper.",
            "Give me the implementation details.",
            "Be specific and technically accurate.",
            "Speak to an expert audience.",
            "Don't dumb this down. I know this domain.",
        ],
        negative_prompts=[
            "Explain this to a complete beginner.",
            "Use simple language. No jargon.",
            "Explain it like I'm 5.",
            "Make it accessible to non-technical people.",
            "Avoid technical terms.",
            "Start from the basics and build up slowly.",
        ],
        default_scale=0.7,
    ),

    "cautious": Behavior(
        name="cautious",
        description="More careful, considered responses. Surfaces risks and edge cases.",
        what_it_does=(
            "The model adds appropriate caveats, surfaces risks, and avoids "
            "overconfident advice. Useful for safety-sensitive domains like "
            "medical, legal, or financial contexts."
        ),
        category="safety",
        when_to_use=(
            "Use when you want the model to be appropriately careful rather than "
            "confidently wrong. Good for safety-sensitive applications."
        ),
        positive_prompts=[
            "What are the risks here?",
            "Be careful and consider the edge cases.",
            "What could go wrong if I follow this advice?",
            "Add appropriate caveats and warnings.",
            "Don't give overconfident advice.",
            "Acknowledge what you don't know.",
        ],
        negative_prompts=[
            "Just tell me what to do. No caveats.",
            "Be confident. Ignore the edge cases.",
            "Assume everything goes right.",
            "Skip the warnings. Just give the answer.",
            "Be decisive. Don't hedge.",
            "I don't need disclaimers. Just answer.",
        ],
        default_scale=0.7,
    ),

}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_behavior(name: str) -> Behavior | None:
    """Return a behavior by name, or None if not found."""
    return CATALOG.get(name)


def list_behaviors() -> list[Behavior]:
    """Return all behaviors, grouped by category."""
    return list(CATALOG.values())
