import openai
import json
from tqdm import tqdm
import os 

openai.api_key = os.env('OPENAI_API_KEY')

if __name__ == "__main__":
    with open('processed/summary_origin/corpus.jsonl', 'r', encoding='utf-8') as f:
        summary_data = [json.loads(l) for l in f]

    print("## Received summary data: ", len(summary_data), summary_data[0])

    prompt = """#  요구사항
- 특허 검색시스템에서 사용될 수 있는 질문을 형성
- 하나의 문서의 세부적인 내용을 확인하기 위한 용도가 아닌, 조건을 만족하는 여러가지 특허 문서들을 찾기 위한 형태로 질문할 것
- 질문 외의 내용은 형성하지 않을 것
- 문서당 5개의 질문을 형성할 것. 내용이 충분하지 않다면 적은 수를 만들 것
- 문서에서 추출한 키워드를 질문에 넣지 말 것
- 문서에서 핵심 키워드들을 뽑아낸 후 유의어로 변경한 것을 활용해서 검색 질문을 만들 것
- 입력 문서내에서 나오는 영어, 특수용어, 단어를 한글로 쉽게 풀어서 사용할 것

예시)
# 입력 문서:
본 발명은 센서 성능 검증 장치 및 그 제어 방법에 관한 것이다. 본 발명의 실시 예에 따른 센서 성능 검증 장치는, 센서의 감지대상이 되는 시험 피검체; 상기 시험 피검체를 회전시키기 위한 회전력 발생부; 상기 센서가 장착되기 위한 센서 장착부; 및 상기 센서 및 시험 피검체의 상대적인 위치를 조절하기 위한 구동부;를 포함할 수 있다. 본 발명의 실시 예에 따른 센서 성능 검증 장치의 제어 방법은, 시험 피검체를 기준 위치로 정렬하는 단계; 제 1 구동부가 구동하여, 센서를 상기 시험 피검체에 대하여 기 설정된 기준 거리로 배치시키는 단계; 상기 시험 피검체를 회전시키는 단계; 상기 시험 피검체가 회전되는 동안, 상기 센서에 감지되는 신호를 획득하는 단계; 및 상기 제 1 구동부가 구동하여, 상기 센서 및 시험 피검체 사이의 거리를 단계적으로 변화시키면서, 위 과정을 반복하는 단계;를 포함할 수 있다.본 발명에 따르면, 하나의 장치를 이용하여, 피검체 크기, 피검체와의 거리 및 피검체의 속도에 따른 센서의 모든 특성을 측정하여, 센서의 모든 성능을 편리하게 검증할 수 있다.

#  출력
키워드 : 센서 성능 검증 장치, 제어 방법, 회전, 센서에 감지되는 신호를 획득, 피검체 크기, 피검체와의 거리, 피검체의 속도
유의어 변경 : 감지기 성능 검증 장치, 통제 방식, 회전, 감지기에 인식되는 시그널을 획득, 검사하는 대상의 크기, 검사 대상과의 거리, 검사 대상의 빠르기

- 회전속도를 자동으로 변경하며 감지기 이상 검출
- 감지기 고장을 검출을 위한 자동 이동 회전치차 거리에 따른 감지기의 출력 측정
- 가속도 센서 또는 진동 센서의 이상을 검출 위한 감지기의 출력을 측정
- 터빈, 펌프, 회전기기 등에 적용되는 감지기의 이상을 감시

# 입력 문서:
    """
    # model = 'gpt-3.5-turbo'
    model = 'gpt-4'

    results = []
    for instance in tqdm(summary_data):
        # gpt-4, gpt-3.5-turbo
        description = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": instance['text']}
            ],
            temperature=1
        )

        response = description['choices'][0]['message']

        results.append({
            "doc_id": instance['_id'],
            "input": instance['text'],
            "generated_text": response,
            "metadata": {"input_type": 'summary', "model": model, "response": description}
        })

    print(f"Processing Done - total: {len(results)}")

    os.makedirs('generated',exist_ok=True)
    with open('generated/generated_text_summary_corpus.json','w',encoding='utf-8') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))