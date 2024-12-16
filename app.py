# 기본 라이브러리
import os
import json
import time
from collections import defaultdict

# 외부 라이브러리
import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px

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
    except Exception as e:
        st.error(f"API 키 로드 중 오류 발생: {str(e)}")
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
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "claude-3-opus-20240229",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 100,
                "temperature": 1
            }
        elif model == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "contents": [{
                    "parts":[{
                        "text": prompt
                    }]
                }]
            }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Handle different API response formats
        if model == "openai":
            return {"choices": [{"text": response.json()["choices"][0]["message"]["content"]}]}
        elif model == "claude":
            return {"choices": [{"text": response.json()["content"][0]["text"]}]}
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
        st.error(f"프롬프트 준비 중 오��� 발생: {str(e)}")
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

# 결과를 표시할 컨테이너 생성
result_container = st.container()

# 진행 상황을 표시할 컨테이너 생성
progress_container = st.container()

# 평가 버튼
if st.button("평가 시작"):
    st.session_state.results = {}
    api_keys = get_api_keys()
    
    if not api_keys:
        st.warning("API 키 설정을 확인해주세요")
    else:
        try:
            # Gemini 평가
            with progress_container:
                st.subheader("Gemini 평가 진행중...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(10):
                    status_text.text(f"질문 {i+1}/10 처리중...")
                    progress_bar.progress((i + 1) * 10)
                    time.sleep(0.1)
                
                st.session_state.results["Gemini"] = evaluate_model("gemini", api_keys["gemini"])
                st.success("Gemini 평가 완료!")

            # Claude 평가
            with progress_container:
                st.subheader("Claude 평가 진행중...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(10):
                    status_text.text(f"질문 {i+1}/10 처리중...")
                    progress_bar.progress((i + 1) * 10)
                    time.sleep(0.1)
                
                st.session_state.results["Claude"] = evaluate_model("claude", api_keys["claude"])
                st.success("Claude 평가 완료!")

            # OpenAI(ChatGPT) 평가
            with progress_container:
                st.subheader("ChatGPT (GPT-4) 평가 진행중...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(10):
                    status_text.text(f"질문 {i+1}/10 처리중...")
                    progress_bar.progress((i + 1) * 10)
                    time.sleep(0.1)
                
                st.session_state.results["ChatGPT"] = evaluate_model("openai", api_keys["openai"])
                st.success("ChatGPT 평가 완료!")

            # 최종 결과 표시
            with result_container:
                st.subheader("평가 결과")
                
                # 각 모델별 결과를 탭으로 구분하여 표시 (순서 변경)
                tabs = st.tabs(["Gemini", "Claude", "ChatGPT"])
                
                # 결과 표시 순서도 변경
                models_order = ["Gemini", "Claude", "ChatGPT"]
                for idx, (model, tab) in enumerate(zip(models_order, tabs)):
                    with tab:
                        st.write(f"### {model} 결과")
                        
                        # 차원별 결과 표시
                        for dimension, data in st.session_state.results[model].items():
                            st.write(f"#### {dimension}")
                            
                            # 템플릿별 결과를 표로 표시
                            for template, results in data.items():
                                st.write(f"**템플릿: {template}**")
                                df = pd.DataFrame(results)
                                st.dataframe(df)
                                
                                # 가능성 시각화
                                fig = px.bar(
                                    x=['Option 1', 'Option 2'],
                                    y=[np.mean([r['likelihood_option1'] for r in results]),
                                       np.mean([r['likelihood_option2'] for r in results])],
                                    title=f"{dimension} - {template} 응답 경향"
                                )
                                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"평가 실패: {str(e)}")
