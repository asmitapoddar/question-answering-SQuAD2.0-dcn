def dbg_print(x):
    print("[debug] ",end='')
    print(x)

def index_convergence_update(index_convergence, s, e):
    if index_convergence is None:
        index_convergence = []
    list_s = list(map(lambda x: x.item(), list(s)))
    list_e = list(map(lambda x: x.item(), list(e)))
    list_zip = list(zip(list_s, list_e))
    index_convergence.append(list_zip)
    return index_convergence

def compute_index_convergence(index_convergence, batch_size):
    dic = index_convergence
    max_iter = len(index_convergence)
    convergence = [None]*batch_size
    convergence_divergence_counter = 0
    for it in range(0,max_iter-1):
        for b in range(batch_size):
            if convergence[b] is None:
                if dic[it+1][b] == dic[it][b]:
                    convergence[b] = it+1
                    for it_prime in range(it+2, max_iter-1):
                        if dic[it_prime][b] != dic[it][b]:
                            convergence_divergence_counter += 1
                            break
    
    dbg_print("%d/%d spans converged and later diverged" %(convergence_divergence_counter, batch_size))
    for val in [None]+list(range(1, max_iter)):
        num = len([v for v in convergence if v==val])
        if val == None:
            dbg_print("%d/%d spans never converged" %(num, batch_size))
        else:
            dbg_print("%d/%d spans converged after %d steps" % (num, batch_size, val))
    return convergence
