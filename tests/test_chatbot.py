import unittest
from chatbot import SupportChatbot


class TestSupportChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = SupportChatbot("data/knowledge_base.json")

    def test_manual_question(self):
        answer, score, matched = self.bot.get_answer("Where do I get the user guide?")
        self.assertIsNotNone(matched)
        self.assertGreater(score, 0)
        self.assertIn("official", answer.lower())

    def test_unknown_question_uses_fallback(self):
        answer, score, matched = self.bot.get_answer("Can you calculate the distance to Mars?")
        self.assertIsNone(matched)
        self.assertIn("not confident", answer.lower())

    def test_empty_question(self):
        answer, score, matched = self.bot.get_answer("   ")
        self.assertIsNone(matched)
        self.assertIn("enter a question", answer.lower())


if __name__ == "__main__":
    unittest.main()
