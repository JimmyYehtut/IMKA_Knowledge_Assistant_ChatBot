# [TEST] Assessment of IMKA

1) `questions.json` contains 70 question and answers pairing on `ABB_Manual_for_Induction_Motors_and_Generators_EN.pdf` targets efficacy of various RAG techniques such as Chunking, Image Captioning & NER.

2) `run-questions-imka.py` answers `questions.json` by calling the running IMKA backend API (localhost) and saves to `/output/answers-imka.json`

3) `evaluate-ragas.py` puts `/output/answers-imka.json` through a RAGAS evaluation with `--luna` (primary judge, frontier model `gpt-5.6-luna`) and/or `--qwen` (secondary judge, qwen3.8:27b); both judges' scores go to `ragas-imka.json`

4) `evaluate-retrieval.py` puts `/output/answers-imka.json` through a retrieval evaluation - Hit Rate@K and MRR (Mean Reciprocal Rank), results to `retrieval-imka.json`

5) `summary-imka.ipynb` charts and plots results from `answers-imka.json`, `ragas-imka.json` and `retrieval-imka.json`; RAGAS charts are drawn for each judge that has scored all 70 questions, plus a judge comparison when both have


###  Full Run of IMKA Assesment

Start Qdrant Docker & Backend 
```bash
cd backend
# Start Qdrant
docker compose up -d
# OPTIONAL: create python virtual environment
python -m venv .venv
# start virtual environment
.venv\Scripts\Activate
# Start Backend
uvicorn api.main:app --reload --port 8000
```

TEST pipeline (run in a new terminal)
```bash
# navigate to test folder
cd test
# OPTIONAL: create python virtual environment
python -m venv .venv
# start virtual environment
.venv\Scripts\Activate
# install packages
pip install -r requirements.txt
# IMKA login for run-questions-imka.py: copy, then fill in IMKA_EMAIL / IMKA_PASSWORD
# (an account that exists in the IMKA UI)
copy .env.example .env

# OPTIONAL: Generate answers using gpt-5.6-luna without RAG
python run-questions-luna.py
# OPTIONAL: Generate answers using qwen3.8:27b without RAG
python run-questions-qwen.py

# Generate answers through IMKA — calls the running backend at http://localhost:8000,
# which uses its configured LLM + Qdrant (each question is saved as a chat in that account)
python run-questions-imka.py

# Evaluate RAGAS (primary judge)
python evaluate-ragas.py --luna --concurrent 4 --redo
# OPTIONAL: Evaluate RAGAS with the secondary judge
python evaluate-ragas.py --qwen --max-jobs 18 --redo
# Evaluate Retrieval
python evaluate-retrieval.py
```
`summary-imka.ipynb` Run All
