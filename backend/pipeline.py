import os
import gdown

ready = False

def init():
    gdown.download(id='103DgLujAKKLlfETz-rgDO0-ibvQh7evQ', output='roboflow-model/best.pt')    # andrà prob spostato in dockerfile
    ready = True

def start(image, team1, team2):
    
    # translate team codes
    team_dict = { 'Juventus': 'sqjvnts', 'Fiorentina': 'sqfrntn', 'Inter': 'sqntrxx', 'Milan': 'sqmlnxx', 'Napoli': 'sqnplxx', 'Roma': 'sqrmxxx' }
    team1_code = team_dict[team1]
    team2_code = team_dict[team2]

    # setup env var
    HOME = os.getcwd()
    print(HOME)

    return image