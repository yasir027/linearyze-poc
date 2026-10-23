from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.text import MESSAGES

BOOTH_CONTRACTS = (
    "Booth 12 contract. Exhibitor: Sample Trading Co. Booth 12 payment is due within 30 days of signing. "
    "Booth 12 is a corner unit of 3 by 4 metres.\n\n"
    "Booth 21 contract. Exhibitor: Demo Foods LLC. Booth 21 payment is due within 45 days of signing."
)


def _client() -> TestClient:
    return TestClient(create_app(Settings(_env_file=None)))


def _upload(client: TestClient, name: str, text: str):
    return client.post("/documents", files={"file": (name, text.encode("utf-8"), "text/plain")})


def test_answer_quotes_the_source_verbatim():
    client = _client()
    assert _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS).status_code == 201

    body = client.post("/query", json={"question": "What are the payment terms for booth 12?"}).json()

    assert body["outcome"] == "B"
    assert body["mode"] == "extractive"
    assert body["quotes"], "an answer must carry at least one quote"
    for quote in body["quotes"]:
        assert quote["text"] in BOOTH_CONTRACTS
    assert any("Booth 12 payment is due within 30 days" in q["text"] for q in body["quotes"])


def test_upload_is_idempotent():
    client = _client()
    first = _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS).json()
    second = _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS).json()
    assert first["id"] == second["id"]
    assert first["passage_count"] == second["passage_count"]


def test_number_in_question_must_appear_in_passage():
    client = _client()
    _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS)
    body = client.post("/query", json={"question": "What is the payment term for booth 99?"}).json()
    assert body["outcome"] == "E"
    assert body["message"] == MESSAGES["M1"]["en"]


def test_unrelated_question_returns_not_found():
    client = _client()
    _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS)
    body = client.post("/query", json={"question": "What is the speed of neutrons in a vacuum?"}).json()
    assert body["outcome"] == "E"
    assert body["quotes"] == []


def test_payment_request_is_refused():
    client = _client()
    _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS)
    body = client.post("/query", json={"question": "pay invoice 4411 now"}).json()
    assert body["outcome"] == "OT"
    assert body["message"] == MESSAGES["M10007"]["en"]


def test_arabic_question_gets_arabic_not_found_message():
    client = _client()
    _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS)
    body = client.post("/query", json={"question": "ما هي سرعة الضوء في الفراغ؟"}).json()
    assert body["outcome"] == "E"
    assert body["message"] == MESSAGES["M1"]["ar"]


def test_unsupported_file_type_is_rejected():
    client = _client()
    response = client.post("/documents", files={"file": ("photo.png", b"\x89PNG", "image/png")})
    assert response.status_code == 415


def test_empty_question_is_rejected():
    client = _client()
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_document_can_be_downloaded_byte_for_byte():
    client = _client()
    uploaded = _upload(client, "booth_contracts.txt", BOOTH_CONTRACTS).json()
    response = client.get(f"/documents/{uploaded['id']}/download")
    assert response.status_code == 200
    assert response.content == BOOTH_CONTRACTS.encode("utf-8")
