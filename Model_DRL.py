import keras.models as ag
import numpy as np
import tensorflow as tf
import tensorflow as tf_agents
from markdown_it.presets.commonmark import make
from Evaluation import evaluation

def Model_DRL(Train_Data, Train_Target, Test_Data, Test_Target):
    env = make('MyClassificationEnvironment-v0')
    # Define the observation and action specifications
    observation_spec = tf_agents.specs.TensorSpec(shape=(1,), dtype=np.int32)
    action_spec = tf_agents.specs.BoundedTensorSpec(shape=(), dtype=np.int32, minimum=0, maximum=9)
    # Create the DQN agent
    q_network = ag.MyQNetwork(observation_spec)
    policy = tf_agents.policies.QPolicy(q_network)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_step_counter = tf.Variable(0)

    agent = tf_agents.DqnAgent(
        environment=env,
        q_network=q_network,
        optimizer=optimizer,
        td_errors_loss_fn=tf_agents.losses.huber_loss,
        train_step_counter=train_step_counter,
        exploration_policy=policy,
        collection_policy=policy,
        actor_fc_layers=(256, 256),
        critic_fc_layers=(256, 256),
        epsilon_greedy=0.1,
        target_update_tau=0.001,
        target_update_period=1,
        gamma=0.99,
        gradient_steps=1000,
        debug_summaries=False,
        n_step_bootstrapped=1,
        reward_scale_factor=1.0)

    # Collect some training data
    agent.collect_data(num_episodes=10)
    # Train the agent
    agent.train(Train_Data, Train_Target,num_iterations=200)
    pred = agent.predict(Test_Data)
    Eval = evaluation(pred,Test_Target)
    return Eval