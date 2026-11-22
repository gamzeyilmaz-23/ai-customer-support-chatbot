"""FAQ retrieval chatbot using TF-IDF and cosine similarity."""

import json
from pathlib import Path
from typing import Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SupportChatbot:
    def __init__(self, knowledge_base_path: str, threshold: float = 0.22):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.threshold = threshold
        self.faqs = self._load_faqs()
        if not self.faqs:
            raise ValueError("The knowledge base is empty.")

        self.questions = [item["question"] for item in self.faqs]
        self.search_texts = [
            item["question"] + " " + " ".join(item.get("keywords", []))
            for item in self.faqs
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.question_matrix = self.vectorizer.fit_transform(self.search_texts)

    def _load_faqs(self):
        with self.knowledge_base_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Knowledge base must be a JSON list.")
        for item in data:
            if not all(key in item for key in ("question", "answer")):
                raise ValueError("Each FAQ must contain 'question' and 'answer'.")
        return data

    def get_answer(self, user_question: str) -> Tuple[str, float, Optional[str]]:
        question = user_question.strip()
        if not question:
            return "Please enter a question so I can help.", 0.0, None

        user_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(user_vector, self.question_matrix)[0]
        best_index = int(scores.argmax())
        best_score = float(scores[best_index])

        if best_score < self.threshold:
            return (
                "I'm not confident that I understand your question. "
                "Please rephrase it or contact the relevant customer support team. "
                "For product-specific instructions, check the official product manual.",
                best_score,
                None,
            )

        return self.faqs[best_index]["answer"], best_score, self.questions[best_index]


if __name__ == "__main__":
    bot = SupportChatbot("data/knowledge_base.json")
    print("Customer Support Chatbot (type 'quit' to exit)")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        answer, score, matched = bot.get_answer(question)
        print(f"Bot: {answer}")
        if matched:
            print(f"[Matched FAQ: {matched} | Similarity: {score:.2f}]")
