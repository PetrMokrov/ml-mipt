
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    # s = `state` 
    # a = `action`
    next_states = mdp.get_next_states(state, action) # set of s'
    q_s_a = 0
    for s_tch in next_states:
        p = mdp.get_transition_prob(state, action, s_tch)
        r = mdp.get_reward(state, action, s_tch)
        v = state_values[s_tch]
        q_s_a += p * (r + gamma*v)

    return q_s_a
