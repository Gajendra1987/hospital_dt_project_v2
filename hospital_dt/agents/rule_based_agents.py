def icu_agent_rule(obs):
    return {'allocate_icu': obs['icu_free'] > 0 and obs['icu_wait'] > 0}
def ot_agent_rule(obs):
    return {'allocate_ot': obs['ot_free'] > 0 and obs['ot_wait'] > 0}
def staff_agent_rule(obs):
    if obs['ed_wait'] > 5:
        return {'reassign_staff': True}
    return {'reassign_staff': False}
def merge_actions(*actions):
    res = {}
    for a in actions:
        res.update(a)
    return res
