class ConversationManager:
    """
    Rule-based multi-turn conversation manager.

    Keeps the current topic and rewrites follow-up questions
    into context-aware retrieval queries.
    """

    FOLLOW_UP_CUES = [
        "那",
        "如果",
        "還是",
        "仍然",
        "也",
        "呢",
        "這樣",
        "它",
        "這個",
        "那個",
        "還有",
    ]

    def __init__(self):
        self.topic_query = None
        self.history = []

    def is_follow_up(self, query):
        query = query.strip()

        if self.topic_query is None:
            return False

        return (
            len(query) <= 18
            or any(cue in query for cue in self.FOLLOW_UP_CUES)
        )

    def prepare_query(self, user_query):
        user_query = user_query.strip()

        if not user_query:
            raise ValueError("User query cannot be empty.")

        follow_up = self.is_follow_up(user_query)

        if follow_up:
            rewritten_query = (
                f"{self.topic_query}；"
                f"使用者進一步詢問：{user_query}"
            )
        else:
            rewritten_query = user_query
            self.topic_query = user_query

        self.history.append({
            "role": "user",
            "content": user_query,
            "rewritten_query": rewritten_query,
            "is_follow_up": follow_up,
        })

        return rewritten_query

    def add_assistant_message(self, message):
        self.history.append({
            "role": "assistant",
            "content": message,
        })

    def reset(self):
        self.topic_query = None
        self.history = []
