
def manualCardInfo(set_abbr)->dict:
    cardInfo={}
    if set_abbr=='mkm':
        cardInfo[500]={'name':'Crashing Footfalls',
                    'mv':0,
                    'color':16,
                    'type':'S',
                    'rarity':'M'}
        cardInfo[501]={'name':'Drown in the Loch',
                    'mv':2,
                    'color':6,
                    'type':'I',
                    'rarity':'M'} 
        cardInfo[502]={'name':'Evolutionary Leap',
                    'mv':2,
                    'color':16,
                    'type':'E',
                    'rarity':'R'}
        cardInfo[503]={'name':'Field of the Dead',
                    'mv':-1,
                    'color':0,
                    'type':'L',
                    'rarity':'M'}
        cardInfo[504]={'name':'Ghostly Prison',
                    'mv':3,
                    'color':1,
                    'type':'E',
                    'rarity':'M'} 
        cardInfo[505]={'name':'Possibility Storm',
                   'mv':5,
                   'color':8,
                   'type':'E',
                   'rarity':'R'}                 
        cardInfo[506]={'name':'Show and Tell',
                   'mv':3,
                   'color':2,
                   'type':'S',
                   'rarity':'M'} 
        cardInfo[507]={'name':"Smuggler\'s Copter",
                   'mv':2,
                   'color':0,
                   'type':'A',
                   'rarity':'R'}    
    return cardInfo                          