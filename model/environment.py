import torch


def scan_state(table, state):
    print('---------------------------------------------------------------------------')
    is_win = 0

    #Check players
    for i in [-1, 1]:

        # Check Diagonally
        for y in range(table[1]):
            for x in range(table[0]):

                # print('-----------', [x, y], '-----------')

                check = True
                for j in range(5):
                    if x-(j-2) < 0 or x-(j-2) >= table[0]: check = False; break
                    if y+(j-2) < 0 or y+(j-2) >= table[1]: check = False; break

                    if x == 2 and y == 2:
                        print([x-(j-2), y+(j-2)], state[x-(j-2)][y+(j-2)])

                    if state[x-(j-2)][y+(j-2)] != i:
                        check = False
                    
                    if not check:
                        break
                
                if check: break

                # Another Diagonally
                check = True
                for j in range(5):
                    if x-(j-2) < 0 or x-(j-2) >= table[0]: check = False; break
                    if y-(j-2) < 0 or y-(j-2) >= table[1]: check = False; break

                    if x == 2 and y == 2:
                        print([x-(j-2), y-(j-2)], state[x-(j-2)][y-(j-2)])

                    if state[x-(j-2)][y-(j-2)] != i:
                        check = False
                    
                    if not check:
                        break
                
                if check: break
            
            if check: break

        if check:
            is_win = i
            break

    return is_win


def step(table, state, AoD, action):

    table_width, table_height = table

    player = 0
    if AoD == 0:
        player = -1
    else:
        player = 1
    
    action = [int(action / table_width), action % table_width]
    if state[action[0]][action[1]] == 0:
        state[action[0]][action[1]] = player
    else:
         return state, -0.1, False
    
    done = True if scan_state(table, state) != 0 else False
    return state, 1 if done else 0, done

state = torch.zeros(15, 15).numpy()
state, reward, done = step((15, 15), state, 0, 0)
state, reward, done = step((15, 15), state, 0, 16)
state, reward, done = step((15, 15), state, 0, 32)
state, reward, done = step((15, 15), state, 0, 48)
state, reward, done = step((15, 15), state, 0, 64)

print(state, done)
