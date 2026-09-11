import pytest

from pcnn4.human_pref import is_dupe, load_groups
from pcnn4.human_sort import (TIER_RANK, compute_from_state, gen_lists,
                              pcreview_tier)


@pytest.fixture(scope='module')
def lists():
    return gen_lists()


def test_list_sizes(lists):
    groups = load_groups()
    assert sorted(lists) == [1, 2, 3, 4, 5, 6, 7]
    for pc, items in lists.items():
        expected = [k for k in groups[pc]
                    if not (pc == 2 and is_dupe(k))]
        assert len(items) == len(expected)
        assert {it['k'] for it in items} == set(expected)
    assert sum(len(v) for v in lists.values()) == 463


def test_default_order_tiers(lists):
    # PCs 2-6 sort by PCReview tier first; within a tier V* descending
    for pc in (2, 3, 4, 5, 6):
        items = lists[pc]
        ranks = [TIER_RANK[it['tier']] for it in items]
        assert ranks == sorted(ranks)
        for a, b in zip(items, items[1:]):
            if a['tier'] == b['tier']:
                assert a['vstar'] >= b['vstar']
    # PCs 1/7 have no PCReview judgement: pure V* descending
    for pc in (1, 7):
        assert all(it['tier'] is None for it in lists[pc])
        vs = [it['vstar'] for it in lists[pc]]
        assert vs == sorted(vs, reverse=True)


def test_pcreview_tier_spotchecks():
    # PC3 receives a single piece: T is top, S/Z bad (pcname._tier3rd)
    assert pcreview_tier('T', 3) == 'top'
    assert pcreview_tier('S', 3) == 'bad'
    assert pcreview_tier('O', 3) == 'mid'
    assert pcreview_tier('IJLOSTZ', 1) is None


def test_compute_chain(lists):
    pc3 = [it['k'] for it in lists[3]]
    state = {3: {'order': pc3, 'gaps': [1, 2, 1, 5, 1, 20]}}
    results = compute_from_state(state, lists)
    w = results[3]['weights']
    assert w[pc3[0]] == 100.0
    assert w[pc3[1]] == 100.0
    assert w[pc3[2]] == 50.0
    assert w[pc3[3]] == 50.0
    assert w[pc3[4]] == 10.0
    assert w[pc3[6]] == 0.5
    assert results[3]['answered'] == 7
    # untouched pcs come back all-None but complete
    assert results[4]['answered'] == 0
    assert all(v is None for v in results[4]['weights'].values())


def test_compute_includes_pc2_dupes_as_none(lists):
    state = {2: {'order': [it['k'] for it in lists[2]],
                 'gaps': [1] * (len(lists[2]) - 1)}}
    results = compute_from_state(state, lists)
    w = results[2]['weights']
    assert len(w) == 140                       # 35 ranked + 105 manual dupes
    ranked = [v for v in w.values() if v is not None]
    assert len(ranked) == 35
    dupes_none = [k for k, v in w.items() if v is None]
    assert all(is_dupe(k) for k in dupes_none)


def test_compute_rejects_stale_export(lists):
    state = {3: {'order': ['T', 'I'], 'gaps': [1]}}
    with pytest.raises(ValueError):
        compute_from_state(state, lists)
