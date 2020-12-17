import os
import pickle

from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from env2 import RLCar


def train():
    env = RLCar()
        
    # Cria um dicionário vazio
    Q = {}
    
    # Inicialização da tabela Q - com 3 estados possíveis (esquerda, frente e direita)
    # relacionado às 3 possíveis ações (11-Frente, 10 Esquerda e 01 Direita.)
    for i in range (0, 10):
        for j in range (0, 10):
            for k in range (0, 10):
                Q[(i/2, j/2, k/2, 1, 1)] = -500000
                Q[(i/2, j/2, k/2, 1, 0)] = -500000
                Q[(i/2, j/2, k/2, 0, 1)] = -500000
    
    alpha = 0.6         # Taxa de aprendizado
    y = .95             # Fator de desconto
    e = 0.5             # Termo que define a probabilidade de tomar ação randômica
    num_episodes = 1000  # Numero de episódios
    num_steps = 10000   # Número de passos dentro de cada episódio
    
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []

    # Definição da velocidade de acionamento dos motores
    SPEED = 1
    
    # Cria um vetor para armazenar as recomepensas
    total_reward = np.zeros(num_episodes)

    for i in range(num_episodes):
        # Reset environment and get first new observation
        env.reset()
        
        rAll = 0
        s = env.readSensor()
        print(f"sensor value episode: {s}")
        
        #Aqui, preciso discretizar as leituras em intervalos condizentes com o Q
        # e guardo como variáveis "s", que representam os estados
        state = np.zeros(3)
        print(f"state value episode: {state}")
        
        # Crio uma condição para eliminar valores errados vindos do sensor
        if (np.abs(s['proxy_sensor'][0]) > 10 or np.abs(s['proxy_sensor'][1]) > 10 or np.abs(s['proxy_sensor'][2] > 10)):
                s['proxy_sensor'][0] = 0
                s['proxy_sensor'][1] = 0
                s['proxy_sensor'][2] = 0
             
        state[0] = int(s['proxy_sensor'][0]*2) / 2
        state[1] = int(s['proxy_sensor'][1]*2) / 2
        state[2] = int(s['proxy_sensor'][2]*2) / 2
        
        
        # Defino aqui a minha tabela         
        Q_values    = [ Q[(state[0], state[1], state[2], 1,1)] , Q[(state[0], state[1], state[2], 1,0)], Q[(state[0], state[1], state[2], 0,1)]]
        acts        = [[1,1] , [1,0] , [0,1]]
        keys        = [(state[0], state[1], state[2], 1,1) , (state[0], state[1], state[2], 1,0), (state[0], state[1], state[2], 0,1)]
        
        # Seleciona a ação que maximiza o vetor Q
        max_id = np.argmax(Q_values)
        act = acts[max_id]
        
        if (np.random.rand() < e) or (Q_values[0] == Q_values[1] == Q_values[2]):
            max_id  = np.random.randint(0,3)
            act     = acts[max_id]
        
        
        for j in range(num_steps):
            # print ("Step numero: ", j)
            
            # Aplica o passo no ambiente e adquire as recompensas e novos estados            
            s, r = env.step([SPEED*act[0], SPEED*act[1]])
            print(f"sensor value step: {s}")
            print(f"reward value step: {r}")
            r = r['proxy_sensor'] + r['light_sensor']
            # print("Reward: ", r)
            total_reward[i] += r 
            rAll += r
            
            
            
            if (np.abs(s['proxy_sensor'][0]) > 10 or np.abs(s['proxy_sensor'][1]) > 10 or np.abs(s['proxy_sensor'][2] > 10)):
                s['proxy_sensor'][0] = 0
                s['proxy_sensor'][1] = 0
                s['proxy_sensor'][2] = 0
                 
            
            
            
            # Discretizo os novos estados
            state_next = np.zeros(3)
            state_next[0] = round(s['proxy_sensor'][0]*2) / 2
            state_next[1] = round(s['proxy_sensor'][1]*2) / 2
            state_next[2] = round(s['proxy_sensor'][2]*2) / 2
            
            # Com base nos novos estados, construo a tabela Q
            Q_values_next = [ Q[(state_next[0], state_next[1], state_next[2], 1,1)] , Q[(state_next[0], state_next[1], state_next[2], 1,0)], Q[(state_next[0], state_next[1], state_next[2], 0,1)]]
            acts_next   = [[1,1] , [1,0] , [0,1]]
            keys_next   = [(state_next[0], state_next[1], state_next[2], 1,1) , (state_next[0], state_next[1], state_next[2], 1,0), (state_next[0], state_next[1], state_next[2], 0,1)]
            
            
            import pdb;pdb.set_trace()
            # Seleciona a ação que maximiza o vetor Q_next
            max_id_next = np.argmax(Q_values_next)
            act_next = acts[max_id_next]
            
            if (np.random.rand() < e) or (Q_values_next[0] == Q_values_next[1] == Q_values_next[2]):
                max_id_next  = np.random.randint(0,3)
                act_next     = acts[max_id_next]
                
            # Calculo da equação do SARSA
            Q[keys[max_id]] =  Q[keys[max_id]] + alpha * (r + y * Q_values_next[max_id_next] - Q[keys[max_id]])
            
            # Atualiza os valores
            Q_values    = Q_values_next
            acts        = acts_next
            keys        = keys_next
            max_id      = max_id_next
            act         = act_next
            
            jList.append(j)
            rList.append(rAll)
          
            
            
        print("reward: ", rAll)                
        plt.plot(rList)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        train()
    except KeyboardInterrupt:
        print('Exiting.')