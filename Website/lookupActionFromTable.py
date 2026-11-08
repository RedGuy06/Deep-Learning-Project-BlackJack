"""Basic-strategy lookup from DeepLearningLookupTables.xlsx."""
from pathlib import Path

TABLE_PATH = Path(__file__).resolve().parent / "DeepLearningLookupTables.xlsx"
NOT_FOUND = "Action not found"

ACTION_TEXT = {
    'H': 'Hit',
    'S': 'Stand',
    'Dh': 'Double if allowed, otherwise Hit',
    'Ds': 'Double if allowed, otherwise Stand',
    'P': 'Split',
    'Ph': 'Split if double after split is allowed, otherwise Hit',
    'Pd': 'Split if double after split is allowed, otherwise double and hit',
    'Rh': 'Surrender if allowed, otherwise Hit',
    'Rs': 'Surrender if allowed, otherwise Stand',
}

_tables = None


def load_tables(path=TABLE_PATH):
    """Read every sheet once: {sheet name: DataFrame}."""
    import pandas as pd
    return pd.read_excel(path, sheet_name=None)


def hand_value(hand):
    """Best total for a hand of ints and 'A'. Returns (total, is_soft)."""
    total = sum(1 if card == 'A' else int(card) for card in hand)
    if 'A' in hand and total + 10 <= 21:
        return total + 10, True
    return total, False


def getActionFromLookup(PlayerHand, DealerCard, PlayerValue=None, tables=None):
    """Return the basic-strategy play as text.

    PlayerValue is accepted so old calls still work; the total is worked out here.
    """
    global _tables
    if tables is None:
        if _tables is None:
            _tables = load_tables()
        tables = _tables

    hand = [c if c == 'A' else int(c) for c in PlayerHand]
    dealer = 'A' if DealerCard == 'A' else int(DealerCard)
    total, soft = hand_value(hand)

    # The Split sheet is keyed by the pair's card, not the total. Pairs of 5s and
    # 10s are played as hard 10 and hard 20 (the Split sheet rows for them differ
    # from standard strategy and from your 'Composition 2-Cards' sheet).
    if len(hand) == 2 and hand[0] == hand[1] and hand[0] not in (5, 10):
        sheet, row = "Total Dependent - Split", hand[0]
    elif soft:
        sheet, row = "Total Dependent - Soft", total
    else:
        sheet, row = "Total Dependent - Hard", total
        if total < 5:
            return ACTION_TEXT['H']

    try:
        code = tables[sheet].set_index("PlayerCard").loc[row, dealer]
    except KeyError:
        return NOT_FOUND
    return ACTION_TEXT.get(str(code).strip(), NOT_FOUND)
