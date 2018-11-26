from frozenlake import FrozenLakeEnv
import numpy as np
import copy

env = FrozenLakeEnv()

#Policy evaluation algorithm
#env: This is an instance of an OpenAI Gym environment, where env.P returns the one-step dynamics.
#policy: This is a 2D numpy array with policy.shape[0] equal to the number of states (env.nS), and policy.shape[1] equal to the number of actions (env.nA).  policy[s][a] returns the probability that the agent takes action a while in state s under the policy.
#gamma: This is the discount rate. It must be a value between 0 and 1, inclusive (default value: 1).
#theta: This is a very small positive number that is used to decide if the estimate has sufficiently converged to the true value function (default value: 1e-8).#V: This is a 1D numpy array with V.shape[0] equal to the number of states (env.nS).  V[s] contains the estimated value of state s under the input policy.
def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v_test = 0
            for a in range(env.nA):
                for p in env.P[s][a]:
                    v_test += policy[s][a] * p[0] * (p[2] + gamma * V[p[1]])
            delta = max(delta, np.absolute(v_test-V[s]))
            V[s] = v_test
        if delta < theta:
            break
    return V

#q: This is a 1D numpy array with q.shape[0] equal to the number of actions (env.nA).  q[a] contains the (estimated) value of state s and action a.
def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, s_next, reward, done in env.P[s][a]:
            q[a] += prob * (reward+gamma*V[s_next])
    return q

def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)
        policy[s][np.argmax(q)] = 1

    return policy

def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.nS, env.nA]) / env.nA
    policy_stable = False
    while policy_stable == False:
        V = policy_evaluation(env, policy, gamma, theta)
        policy_star = policy_improvement(env, V, gamma)
        
        if policy_star.all() == policy.all():
            policy_stable = True
        
        policy = policy_star

    return policy, V

def truncated_policy_evaluation(env, policy, V, max_it=1, gamma=1):
    counter = 0
    while counter < max_it:
        for s in range(env.nS):
            v = 0
            q = q_from_v(env, V, s, gamma)
            for a in range(env.nA):
                v += policy[s][a] * q[a]
            V[s] = v
        counter += 1
    return V

def truncated_policy_iteration(env, max_it=1, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA]) / env.nA

    while True:
        policy = policy_improvement(env, V)
        V_old = copy.copy(V)
        V = truncated_policy_evaluation(env, policy, V, max_it, gamma)
        if max(abs(V-V_old)) < theta:
            break

    return policy, V

def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta, np.absolute(v-V[s]))
        if delta < theta:
            break
    policy = policy_improvement(env, V)
    
    
    return policy, V

def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)

    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta, np.absolute(v-V[s]))
        if delta < theta:
            break
    policy = policy_improvement(env, V)


    return policy, V
