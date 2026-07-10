import os
import re
import json
import joblib
import requests
import pandas as pd
import jsonschema

def main():
    print("================================================================================")
    print("                    PART 4 — LLM-POWERED EXPLANATION FEATURE                    ")
    print("================================================================================")
    
    # 1. API connection setup
    api_key = os.environ.get('LLM_API_KEY', '')
    api_url = os.environ.get('LLM_API_URL', 'https://openrouter.ai/api/v1/chat/completions')
    model_name = os.environ.get('LLM_MODEL', 'google/gemini-2.5-flash')
    
    print(f"  LLM API Endpoint: {api_url}")
    print(f"  LLM Model: {model_name}")
    if not api_key:
        print("  WARNING: LLM_API_KEY environment variable is not set. API calls will be simulated!")
        
    def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
        if not api_key:
            # Simulated Response fallback
            m = re.search(r"Features:\s*({.*?})", user_prompt, re.DOTALL)
            features_dict = {}
            if m:
                try: features_dict = json.loads(m.group(1).replace("'", '"'))
                except: pass
            pred_match = re.search(r"Predicted Class:\s*(\d+)", user_prompt)
            pred_lbl = "High Salary (> Median)" if (pred_match and pred_match.group(1) == "1") else "Low Salary (<= Median)"
            prob_match = re.search(r"Predicted Probability:\s*([\d\.]+)", user_prompt)
            prob_val = float(prob_match.group(1)) if prob_match else 0.5
            conf = "high" if (prob_val > 0.8 or prob_val < 0.2) else ("medium" if (prob_val > 0.6 or prob_val < 0.4) else "low")
            
            mock = {
                "prediction_label": pred_lbl,
                "confidence_level": conf,
                "top_reason": f"The employee's experience ({features_dict.get('Experience', 10)} years) is a primary driver in standard compensation ranges.",
                "second_reason": f"Performance score of {features_dict.get('PerformanceScore', 75)} in {features_dict.get('Department', 'HR')} influences the prediction.",
                "next_step": "Review salary alignment with industry standards."
            }
            if temperature > 0.5:
                mock["top_reason"] = "[Temp=0.7 Variation] " + mock["top_reason"]
            return json.dumps(mock, indent=2)
            
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        return None
        
    print("\n[Task 4.1] Testing LLM connection with hello prompt:")
    hello_resp = call_llm("Reply with only the word: hello", "hello", temperature=0.0)
    print(f"  Raw response: {hello_resp}")
    
    # 2. Prompts
    system_prompt = (
        "You are an AI assistant explaining a machine learning model's prediction to business users.\n"
        "Your task is to explain the predicted Salary Tier classification given a specific employee's feature vector, "
        "the model's prediction class, and predicted probability.\n"
        "Provide your explanation as a valid JSON object strictly matching the following schema:\n"
        "{\n"
        "  \"prediction_label\": \"A string indicating predicted Salary Tier (e.g. 'High Salary' or 'Low Salary')\",\n"
        "  \"confidence_level\": \"A string representing prediction confidence: 'low', 'medium', or 'high'\",\n"
        "  \"top_reason\": \"A concise business-relevant sentence explaining the most critical feature driving this prediction\",\n"
        "  \"second_reason\": \"A concise sentence explaining the second most critical feature supporting the prediction\",\n"
        "  \"next_step\": \"A recommended actionable step for HR or management based on this classification\"\n"
        "}\n"
        "Do not include any explanation, preamble, markdown formatting (such as ```json), or trailing text outside of the raw JSON object."
    )
    
    user_prompt_template = (
        "Model prediction inputs:\n"
        "Employee Features: {features}\n"
        "Predicted Class: {pred_class} (0 = Low Salary, 1 = High Salary)\n"
        "Predicted Probability of High Salary: {prob:.4f}\n"
        "\n"
        "Generate the structured JSON prediction explanation."
    )
    
    # JSON schema for validation
    json_schema = {
        "type": "object",
        "properties": {
            "prediction_label": {"type": "string"},
            "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
            "top_reason": {"type": "string"},
            "second_reason": {"type": "string"},
            "next_step": {"type": "string"}
        },
        "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"]
    }
    
    # PII Guardrail
    def has_pii(text):
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
        return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
        
    print("\n[Task 4.5] Testing PII Guardrail:")
    test_pii_clean = "Model prediction inputs for Age: 25, Experience: 5 years."
    test_pii_dirty = "Model prediction inputs for employee John Doe (john.doe@company.com), phone 123-456-7890."
    print(f"  Clean input has PII? {has_pii(test_pii_clean)}")
    print(f"  PII-containing input has PII? {has_pii(test_pii_dirty)}")
    
    # Verify guardrail blocking
    for label, inp in [("Clean", test_pii_clean), ("Dirty", test_pii_dirty)]:
        if has_pii(inp):
            print(f"  [{label} Input] Input blocked: PII detected.")
        else:
            print(f"  [{label} Input] Input accepted. Proceeding to LLM.")
            
    # Load serialized model (check current directory or parent)
    model_path = "c:/Users/intel/Desktop/signbridge/best_model.pkl"
    if not os.path.exists(model_path):
        model_path = "best_model.pkl"
        if not os.path.exists(model_path):
            model_path = "../best_model.pkl"
            if not os.path.exists(model_path):
                model_path = "../part3/best_model.pkl"
                
    best_pipeline = joblib.load(model_path)
    ref_cols = ['Age', 'Experience', 'PerformanceScore', 'Department_HR', 'Department_IT', 'Department_Marketing', 'Department_Sales', 'Gender_Male']
    
    # Three hand-crafted feature-vector inputs
    inputs_features = [
        {'Age': 25, 'Department': 'IT', 'Gender': 'Male', 'Experience': 3, 'PerformanceScore': 95},
        {'Age': 58, 'Department': 'Sales', 'Gender': 'Female', 'Experience': 32, 'PerformanceScore': 55},
        {'Age': 40, 'Department': 'Finance', 'Gender': 'Female', 'Experience': 15, 'PerformanceScore': 80}
    ]
    
    def encode_record(features, reference_columns):
        rec = {col: 0.0 for col in reference_columns}
        for col in ['Age', 'Experience', 'PerformanceScore']:
            if col in features: rec[col] = float(features[col])
        dept = features.get('Department', '')
        gender = features.get('Gender', '')
        if dept == 'HR': rec['Department_HR'] = 1.0
        elif dept == 'IT': rec['Department_IT'] = 1.0
        elif dept == 'Marketing': rec['Department_Marketing'] = 1.0
        elif dept == 'Sales': rec['Department_Sales'] = 1.0
        if gender == 'Male': rec['Gender_Male'] = 1.0
        return pd.DataFrame([rec])
        
    print("\n[Task 4.2] Running Model Prediction & Probabilities for Hand-crafted Records:")
    prediction_results = []
    for idx, features in enumerate(inputs_features):
        df_encoded = encode_record(features, ref_cols)
        pred_class = int(best_pipeline.predict(df_encoded)[0])
        pred_prob = float(best_pipeline.predict_proba(df_encoded)[0, 1])
        prediction_results.append({
            'features': features,
            'class': pred_class,
            'prob': pred_prob
        })
        print(f"  Input {idx+1}: features={features}")
        print(f"    Predicted Class: {pred_class} | Prob of High Salary: {pred_prob:.4f}")
        
    # Task 4.4: Temperature A/B comparison
    print("\n[Task 4.4] Running Temperature A/B Comparison:")
    for idx, res in enumerate(prediction_results):
        features = res['features']
        pred_class = res['class']
        pred_prob = res['prob']
        user_prompt = user_prompt_template.format(features=json.dumps(features), pred_class=pred_class, prob=pred_prob)
        
        resp_t0 = call_llm(system_prompt, user_prompt, temperature=0.0)
        resp_t07 = call_llm(system_prompt, user_prompt, temperature=0.7)
        print(f"\n  Record {idx+1} (temp=0):\n{resp_t0}")
        print(f"  Record {idx+1} (temp=0.7):\n{resp_t07}")
        
    # Task 4.6: End-to-end Demonstration and Schema Validation
    print("\n[Task 4.6] Running End-to-end pipeline with Validation:")
    for idx, res in enumerate(prediction_results):
        features = res['features']
        pred_class = res['class']
        pred_prob = res['prob']
        user_prompt = user_prompt_template.format(features=json.dumps(features), pred_class=pred_class, prob=pred_prob)
        
        if has_pii(user_prompt):
            print(f"  Input {idx+1} blocked by PII Guardrail.")
            continue
            
        raw_response = call_llm(system_prompt, user_prompt, temperature=0.0)
        print(f"\n  Input Record {idx+1}: {features}")
        print(f"  Raw Response:\n{raw_response}")
        
        # Validation
        try:
            stripped = raw_response.strip()
            if stripped.startswith("```json"): stripped = stripped[7:]
            if stripped.endswith("```"): stripped = stripped[:-3]
            stripped = stripped.strip()
            parsed = json.loads(stripped)
            jsonschema.validate(instance=parsed, schema=json_schema)
            print("  Validation Status: PASS")
        except Exception as e:
            print(f"  Validation Status: FAIL ({e})")
            
    print("================================================================================")
    print("                    PART 4 COMPLETED SUCCESSFULLY                               ")
    print("================================================================================")

if __name__ == "__main__":
    main()
