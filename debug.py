def dbg_print(x):
    print("[debug] ",end='')
    print(x)

def debug_index_convergence_update(debug_index_convergence, s, e):
    if debug_index_convergence is None:
        debug_index_convergence = []
    list_s = list(map(lambda x: x.item(), list(s)))
    list_e = list(map(lambda x: x.item(), list(e)))
    list_zip = list(zip(list_s, list_e))
    debug_index_convergence.append(list_zip)
    return debug_index_convergence

def debug_index_convergence_print(debug_index_convergence, batch_size):
    dic = debug_index_convergence
    max_iter = len(debug_index_convergence)
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
    
