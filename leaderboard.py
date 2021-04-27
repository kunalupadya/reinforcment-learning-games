from firebase import firebase
import json

def put(game, algo, result, params, iters):
    fb = firebase.FirebaseApplication('https://rffl-71346-default-rtdb.firebaseio.com/', None)
    store = {}
    store['algo'] = algo
    store['params'] = params
    store['iters'] = iters
    store['result'] = result
    js = json.dumps(store)
    result = fb.post('/'+game, js)
    print(result)


def get(game):
    fb = firebase.FirebaseApplication('https://rffl-71346-default-rtdb.firebaseio.com/', None)
    resp = fb.get('/' + game, None)
    ret = [json.loads(resp[i]) for i in resp]
    return ret

if __name__ == "__main__":
    print(get("CartPole-v1"))