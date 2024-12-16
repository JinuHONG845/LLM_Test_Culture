import streamlit as st
import json
import requests
import numpy as np
from collections import defaultdict
import os

# 결과를 위한 세션 상태 초기화
if 'results' not in st.session_state:
    st.session_state.results = {}

# API 키 로드 - 에러 처리 포함
def get_api_keys():
    try:
        return {
            "openai": st.secrets["OPENAI_API_KEY"],
            "claude": st.secrets["CLAUDE_API_KEY"],
            "gemini": st.secrets["GEMINI_API_KEY"]
        }
    except Exception:
        return None

# CDEval.json 파일 로드 - 에러 처리 포함
@st.cache_data
def load_questions(filepath="CDEval.json"):
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding='utf-8') as file:
                return json.load(file)
        else:
            # 파일이 없을 경우 샘플 질문으로 대체
            return [{"Question": "Sample question?", 
                    "Option 1": "Option A", 
                    "Option 2": "Option B",
                    "Dimension": "Test"}]
    except Exception as e:
        st.error(f"질문 로드 중 오류 발생: {str(e)}")
        return []

# API 호출을 위한 헬퍼 함수 - 개선된 에러 처리 포함
def call_api(model, prompt, api_key):
    try:
        if not api_key:
            return {"choices": [{"text": "API 키가 제공되지 않았습니다"}]}

        if model == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 1,
                "max_tokens": 100
            }
        elif model == "claude":
            url = "https://api.anthropic.com/v1/complete"
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "claude-instant-1",
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "temperature": 1,
                "max_tokens_to_sample": 100
            }
        elif model == "gemini":
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro/generateContent"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [{"parts":[{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 1,
                    "maxOutputTokens": 100
                }
            }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Handle different API response formats
        if model == "openai":
            return {"choices": [{"text": response.json()["choices"][0]["message"]["content"]}]}
        elif model == "claude":
            return {"choices": [{"text": response.json()["completion"]}]}
        elif model == "gemini":
            return {"choices": [{"text": response.json()["candidates"][0]["content"]["parts"][0]["text"]}]}
            
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed for {model}: {str(e)}")
        return {"choices": [{"text": f"Error: {str(e)}"}]}
    except Exception as e:
        st.error(f"Unexpected error for {model}: {str(e)}")
        return {"choices": [{"text": "Error occurred"}]}

# 질문 프롬프트 준비
def prepare_prompt(question, option1, option2, template="A/B"):
    try:
        if template == "A/B":
            return f"Question: {question}\n(A) {option1}\n(B) {option2}\nAnswer:"
        elif template == "Repeat":
            return f"Question: {question}\n다음 옵션 중 하나만 정확히 반복해서 응답하세요:\n{option1}\n{option2}\nAnswer:"
        elif template == "Compare":
            return f"Question: {question}\n{option1}을(를) {option2}보다 선호하십니까? 예 또는 아니오로만 응답하세요.\nAnswer:"
        else:
            return f"Question: {question}\n(A) {option1}\n(B) {option2}\nAnswer:"
    except Exception as e:
        st.error(f"프롬프트 준비 중 오류 발생: {str(e)}")
        return ""

# 방향성 가능성 계산 - 에러 처리 포함
def calculate_orientation_likelihood(responses, target_option):
    try:
        if not responses:
            return 0
        return sum(1 if response.strip().upper() == target_option.upper() else 0 
                  for response in responses) / len(responses)
    except Exception as e:
        st.error(f"가능성 계산 중 오류 발생: {str(e)}")
        return 0

# 주요 계산 함수
def evaluate_model(model_name, api_key):
    results = defaultdict(lambda: defaultdict(list))
    questions = load_questions()
    
    if not questions:
        st.error("질문이 로드되지 않았습니다")
        return results

    # 진행 상태 바
    progress_bar = st.progress(0)
    total_steps = len(questions[:10]) * 3  # 3개 템플릿
    current_step = 0

    try:
        for question_data in questions[:10]:
            question = question_data.get("Question", "")
            option1 = question_data.get("Option 1", "")
            option2 = question_data.get("Option 2", "")
            dimension = question_data.get("Dimension", "Unknown")
            
            for template in ["A/B", "Repeat", "Compare"]:
                prompt = prepare_prompt(question, option1, option2, template)
                responses = []
                
                # 재시도 메커니즘을 포함한 응답 수집
                for _ in range(3):  # 5회에서 3회로 감소
                    response = call_api(model_name, prompt, api_key)
                    if response and "choices" in response:
                        responses.append(response["choices"][0]["text"])
                
                results[dimension][template].append({
                    "question": question,
                    "responses": responses,
                    "likelihood_option1": calculate_orientation_likelihood(responses, "A"),
                    "likelihood_option2": calculate_orientation_likelihood(responses, "B"),
                })

                current_step += 1
                progress_bar.progress(current_step / total_steps)

    except Exception as e:
        st.error(f"평가 중 오류 발생: {str(e)}")
    finally:
        progress_bar.empty()
    
    return results

# 스트림릿 UI
st.title("LLM 문화적 차원 평가")

# API 키 입력을 위한 사이드바
st.sidebar.title("API 키 입력")
openai_api_key = st.sidebar.text_input("OpenAI API 키", type="password")
claude_api_key = st.sidebar.text_input("Claude API 키", type="password")
gemini_api_key = st.sidebar.text_input("Gemini API 키", type="password")

# 평가 버튼
if st.sidebar.button("평가 시작"):
    st.session_state.results = {}
    
    if not (openai_api_key and claude_api_key and gemini_api_key):
        st.warning("모든 API 키를 입력해주세요")
    else:
        try:
            with st.spinner("OpenAI 평가 중..."):
                st.session_state.results["OpenAI"] = evaluate_model("openai", openai_api_key)
            with st.spinner("Claude 평가 중..."):
                st.session_state.results["Claude"] = evaluate_model("claude", claude_api_key)
            with st.spinner("Gemini 평가 중..."):
                st.session_state.results["Gemini"] = evaluate_model("gemini", gemini_api_key)
            
            st.success("평가 완료!")
            st.write("결과:")
            st.json(st.session_state.results)
        except Exception as e:
            st.error(f"평가 실패: {str(e)}")

# 이전 결과가 있다면 표시
if st.session_state.results:
    st.write("이전 결과:")
    st.json(st.session_state.results)
