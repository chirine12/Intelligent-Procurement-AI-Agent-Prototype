"""Wording of explanations: Groq LLM when available, Python templates otherwise.

The LLM only turns facts into text. It receives the scores, the risk reasons
and the actions already decided by the agent, and returns three texts:
an explanation, a recommendation and a supplier reminder email.

If there is no API key, the call fails, or the answer is not usable, the same
three texts are produced by deterministic templates.
"""

import json
import os

from src import config

REQUIRED_FIELDS = ["explanation", "recommendation", "reminder_email"]

SYSTEM_PROMPT = """You are a procurement analyst assistant in a manufacturing company.
You receive a JSON object with verified facts about one purchase order.

Rules:
- Use ONLY the facts in the JSON. Never invent numbers, names, dates or causes.
- Never change, recompute or question the scores.
- Two different values describe the delay. Mention both in the explanation and keep them apart:
  - "observed_status" is a fact about this order, e.g. "Overdue by 3 day(s)" or "On track". Quote it.
  - "supplier_late_rate_text" is a statistic about the supplier: the share of its past
    deliveries that arrived late. Write it as: "the supplier delivered <value> of its past
    orders late". It describes the supplier's history, not a prediction for this order.
- The actions are already decided. Explain them; do not add, remove or authorise actions.
- Each action has an "authorization" field. When it says "requires approval by ...",
  say that the action is proposed for approval by that role, never that it will be done.
- Plain business English, no markdown.

Return a JSON object with exactly these keys:
- "explanation": 2 to 4 sentences on why this order is at risk.
- "recommendation": 1 to 3 sentences justifying the recommended actions.
- "reminder_email": a short, polite email to the supplier asking to confirm the delivery date.
"""

ACTION_SENTENCES = {
    "monitor": "No action is needed now; keep monitoring the order.",
    "send_supplier_reminder": "Send a reminder asking the supplier to confirm the delivery date.",
    "reschedule_delivery": "Agree a new confirmed delivery date with the supplier.",
    "replace_supplier": "Consider sourcing from a more reliable supplier.",
}


def load_env_file(path=None):
    """Read KEY=VALUE lines from .env into the environment (if not already set)."""
    path = path or config.PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def template_narrative(facts):
    """Deterministic texts built from the same facts the LLM would receive."""
    po, supplier, scores = facts["purchase_order"], facts["supplier"], facts["scores"]
    explanation = (
        f"Purchase order {po['po_id']} for {po['material_description']} from {supplier['name']} "
        f"is ranked #{scores['rank']} with a priority score of {scores['priority_score']} "
        f"and a {scores['risk_level'].lower()} composite risk score of "
        f"{scores['composite_risk_score']}. Observed status: {scores['observed_status']}; "
        f"the supplier delivered {scores['supplier_late_rate_text']} of its past orders late. "
        f"Risk factors: {'; '.join(facts['risk_reasons'])}."
    )
    recommendation = " ".join(
        f"[{action['authorization']}] {ACTION_SENTENCES.get(action['action'], action['action'])} "
        f"Reason: {action['reason']}"
        for action in facts["recommended_actions"]
    )
    reminder_email = (
        f"Dear {supplier['name']} team,\n\n"
        f"Purchase order {po['po_id']} ({po['quantity']:.0f} x {po['material_description']}) "
        f"was promised for {po['promised_delivery_date']}. "
        f"Please confirm the delivery date by return.\n\n"
        f"Kind regards,\nProcurement Team"
    )
    return {"explanation": explanation, "recommendation": recommendation,
            "reminder_email": reminder_email}


def call_groq(facts):
    """Ask the LLM for the three texts. Raises an exception on any problem."""
    from groq import Groq  # imported here so the fallback works without the package

    client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=config.LLM_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", config.DEFAULT_GROQ_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    texts = json.loads(response.choices[0].message.content)

    # "Usable" means: a JSON object with the three expected non-empty texts,
    # and an explanation that states both the supplier late rate and the observed status.
    for field in REQUIRED_FIELDS:
        if not isinstance(texts.get(field), str) or not texts[field].strip():
            raise ValueError(f"LLM response has no usable '{field}'")
    scores, explanation = facts["scores"], texts["explanation"].lower()
    observed = "overdue" if scores["observed_status"] != "On track" else "on track"
    if scores["supplier_late_rate_text"] not in explanation or observed not in explanation:
        raise ValueError("LLM explanation does not state both the supplier late rate and the observed status")
    return {field: texts[field].strip() for field in REQUIRED_FIELDS}


def generate_narrative(facts, use_llm=True):
    """Return the three texts plus where they came from.

    Result keys: explanation, recommendation, reminder_email,
    source ("llm" or "template") and fallback_reason (None when not needed).
    """
    if not use_llm:
        reason = "LLM disabled"
    elif not os.environ.get("GROQ_API_KEY"):
        reason = "GROQ_API_KEY is not set"
    else:
        try:
            return dict(call_groq(facts), source="llm", fallback_reason=None)
        except Exception as error:  # any failure must end in the fallback
            reason = f"LLM call failed: {type(error).__name__}: {error}"
    return dict(template_narrative(facts), source="template", fallback_reason=reason)
