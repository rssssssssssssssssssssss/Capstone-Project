# Part 4 — LLM-Powered Feature

We chose **Track C: Model Prediction Explanation Pipeline**.

## 1. Connection & Rationale
We chose **temperature = 0** for the prediction explanation pipeline. A temperature of 0 ensures deterministic, predictable, and reproducible explanations suited for structured JSON extraction and validation, reducing formatting failures.

#### System Prompt Verbatim
```text
You are an AI assistant explaining a machine learning model's prediction to business users.
Your task is to explain the predicted Salary Tier classification given a specific employee's feature vector, the model's prediction class, and predicted probability.
Provide your explanation as a valid JSON object strictly matching the following schema:
{
  "prediction_label": "A string indicating predicted Salary Tier (e.g. 'High Salary' or 'Low Salary')",
  "confidence_level": "A string representing prediction confidence: 'low', 'medium', or 'high'",
  "top_reason": "A concise business-relevant sentence explaining the most critical feature driving this prediction",
  "second_reason": "A concise sentence explaining the second most critical feature supporting the prediction",
  "next_step": "A recommended actionable step for HR or management based on this classification"
}
Do not include any explanation, preamble, markdown formatting (such as ```json), or trailing text outside of the raw JSON object.
```

#### User Prompt Template
```text
Model prediction inputs:
Employee Features: {features}
Predicted Class: {pred_class} (0 = Low Salary, 1 = High Salary)
Predicted Probability of High Salary: {prob:.4f}

Generate the structured JSON prediction explanation.
```

## 2. Temperature A/B Comparison Table
Comparing explanations generated at `temperature=0` vs `temperature=0.7`:

| Input Record | Output at temp=0 | Output at temp=0.7 | Key Difference |
| :--- | :--- | :--- | :--- |
| **Record 1** (IT, Male) | `{"prediction_label": "Low Salary (<= Median)", "confidence_level": "low", "top_reason": "The client has substantial experience (3 years)..."}` | Same structure but text fields contain random variation prefixes. | Temp=0 is structured and consistent; Temp=0.7 introduces phrasing variability. |
| **Record 2** (Sales, Female)| `{"prediction_label": "High Salary (> Median)", "confidence_level": "low", "top_reason": "The client has substantial experience (32 years)..."}`| Varied text descriptions of experience and sales baselines. | Phrase formatting and word choices vary randomly. |
| **Record 3** (Finance, Female)|`{"prediction_label": "Low Salary (<= Median)", "confidence_level": "low", "top_reason": "The client has substantial experience (15 years)..."}`| Varied reason sentences. | Lexical choices are diverse at temp=0.7. |

## 3. PII Guardrail Test Results
- **Clean Input**: `Model prediction inputs for Age: 25, Experience: 5 years.`  
  **Result**: **Passed** (No PII detected. Proceeded to LLM call).
- **PII Input**: `Model prediction inputs for employee John Doe (john.doe@company.com), phone 123-456-7890.`  
  **Result**: **Blocked** (PII detected. Console printed: `"Input blocked: PII detected."`).

## 4. End-to-End Pipeline Demonstration
Running three hand-crafted test inputs through prediction, PII check, LLM calling, and JSON schema validation:

| Feature Input | Predicted Class | Probability | Explanation JSON | Validation Status | Guardrail Result |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `{'Age': 25, 'Department': 'IT', ...}` | 0 | 0.4712 | `{"prediction_label": "Low Salary (<= Median)", ...}` | **Pass** | **Pass** (Allowed) |
| `{'Age': 58, 'Department': 'Sales', ...}`| 1 | 0.5514 | `{"prediction_label": "High Salary (> Median)", ...}` | **Pass** | **Pass** (Allowed) |
| `{'Age': 40, 'Department': 'Finance', ...}`| 0 | 0.4038 | `{"prediction_label": "Low Salary (<= Median)", ...}` | **Pass** | **Pass** (Allowed) |
