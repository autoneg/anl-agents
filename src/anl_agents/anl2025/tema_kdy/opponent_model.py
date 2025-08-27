from collections import defaultdict

class SimpleOpponentModel:
    def __init__(self):
        self.opponent_bids = []  # 提案履歴を記録
        self.bid_freq = defaultdict(int)  # 各bidの出現回数（任意）

    def update(self, state):
        # 相手の新しい提案を取得
        bid = state.current_offer
        self.opponent_bids.append(bid)
        self.bid_freq[bid] += 1

    def most_frequent_bids(self, top_k=5):
        # 頻出順にソートして上位を返す
        return sorted(self.bid_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]
