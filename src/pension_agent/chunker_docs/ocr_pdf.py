"""
PDF를 CLOVA OCR로 보내는 범용 함수.
- 특정 페이지만 골라서 보내기 가능 (예: doc58의 2,3페이지만)
- 10페이지 넘으면 자동으로 10페이지씩 잘라서 여러 번 요청 후 결과 합침
  (CLOVA OCR 자체가 PDF 1건당 최대 10페이지까지만 지원하기 때문)
"""
import requests
import base64
import uuid
import time
import json
from pypdf import PdfReader, PdfWriter
import io

# ===== 여기 2개만 채우면 됨 =====
SECRET_KEY = "TERCZE5zYW9ZVkxxU1FOaFZkekVvS0xpQ2ZPaE5telA="
INVOKE_URL = "https://yfxd8pzejs.apigw.ntruss.com/custom/v1/56905/ae2433cf6b3c1b048b82afdff9ec787bf7dc2582bbd77da92e49cab58d382e18/general"
# ================================


def extract_pages_to_bytes(pdf_path, page_numbers):
    """
    원본 PDF에서 원하는 페이지만 뽑아 새로운 '임시 PDF'를 메모리 상에 만듦.
    page_numbers: 1부터 시작하는 페이지 번호 리스트 (예: [2, 3])
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for p in page_numbers:
        writer.add_page(reader.pages[p - 1])  # pypdf는 0부터 시작이라 -1

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def call_clova_ocr_bytes(pdf_bytes, name="doc"):
    """PDF 바이트 데이터를 CLOVA OCR로 전송"""
    image_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    request_body = {
        "version": "V2",
        "requestId": str(uuid.uuid4()),
        "timestamp": int(round(time.time() * 1000)),
        "lang": "ko",
        "enableTableDetection": True,
        "images": [
            {
                "format": "pdf",
                "name": name,
                "data": image_base64,
            }
        ],
    }
    headers = {"X-OCR-SECRET": SECRET_KEY, "Content-Type": "application/json"}
    response = requests.post(INVOKE_URL, headers=headers, data=json.dumps(request_body))
    return response


def ocr_pdf(pdf_path, page_numbers=None):
    """
    PDF를 OCR로 처리하는 메인 함수.
    page_numbers를 안 주면 전체 페이지, 주면 그 페이지들만.
    10페이지 넘으면 자동으로 나눠서 보내고 결과를 하나로 합쳐서 반환.
    반환값: {"images": [...]} 형태 (CLOVA OCR 응답 형식과 동일, 여러 배치를 합친 것)
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    if page_numbers is None:
        page_numbers = list(range(1, total_pages + 1))

    print(f"총 {len(page_numbers)}개 페이지 처리 예정: {page_numbers}")

    all_images_result = []
    # 10페이지씩 묶어서 처리
    for i in range(0, len(page_numbers), 10):
        batch = page_numbers[i:i + 10]
        print(f"  -> {batch} 페이지 배치 전송 중...")

        pdf_bytes = extract_pages_to_bytes(pdf_path, batch)
        response = call_clova_ocr_bytes(pdf_bytes, name=f"batch_{i}")

        if response.status_code != 200:
            print(f"  ⚠️ 에러 발생 (상태코드 {response.status_code}): {response.text[:300]}")
            continue

        result = response.json()
        batch_images = result["images"]

        # 이 배치 안에서의 페이지 순서(pageIndex, 0부터 시작)를,
        # 원본 문서 기준 실제 페이지 번호로 바꿔서 기록해둠
        for img, real_page_num in zip(batch_images, batch):
            img["realPageNumber"] = real_page_num
            all_images_result.append(img)

    return {"images": all_images_result}


if __name__ == "__main__":
    import sys
   
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/doc22.pdf"
    PAGE_NUMBERS = None   # doc22는 5페이지라 10페이지 이하 → 자동분할 로직 안 타도 됨, 전체 그대로
    result = ocr_pdf(pdf_path, page_numbers=PAGE_NUMBERS)

    OUTPUT_PATH = "doc22_ocr_response.json"   # ← 이 줄만 doc53 → doc22로 변경
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


    print(f"\n완료! {len(result['images'])}개 페이지 결과 → {OUTPUT_PATH}")
