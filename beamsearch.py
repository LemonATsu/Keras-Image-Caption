import numpy as np

def beamsearch(model, enc_map, dec_map, img, k=4, max_len=10):

    use_unk = False
    oov = enc_map['<RARE>']
    empty = enc_map['<ST>']
    eos = enc_map['<ED>']

    dead_k = 0 # samples that reached eos
    dead_samples = []
    dead_scores = []
    live_k = 1 # samples that did not yet reached eos
    live_samples = [[empty]]
    live_scores = [0]

    cnt = 0
    while live_k and dead_k < k:
        # for every possible live sample calc prob for every possible label
        lang_input = [live_samples[r][-1] for r in range(0, len(live_samples))]
        img_input  = np.tile(np.array([img]), (len(lang_input),1))
        vhists = np.zeros((len(lang_input), 2187))
        seqs = np.zeros((len(live_samples), 53))

        for r in range(0, len(live_samples)):
            url = unroll(live_samples[r][0])
            idx = len(url)-1
            seqs[r, idx] = 1
            vhists[r, np.array(url)] = 1
            X = [img_input, np.array(lang_input).reshape(-1,1), seqs, vhists]

        probs = model.predict(X)[0]

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:,None] - np.log(probs)
        if not use_unk and oov is not None:
            cand_scores[:,oov] = 1e20
        cand_flat = cand_scores.flatten()

        # find the best (lowest) scores we have from all possible samples and new words
        ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        live_scores = cand_flat[ranks_flat]

        # append the new words to their appropriate live sample
        voc_size = len(probs)
        n_livesample = []
        live_samples = [[unroll(live_samples[r//voc_size])]+[r%voc_size] for r in ranks_flat]

        # live samples that should be dead are...
        zombie = [s[-1] == eos or len(s[0]) >= max_len for s in live_samples]

        # add zombies to the dead
        dead_samples += [s for s,z in zip(live_samples,zombie) if z]  # remove first label == empty
        dead_scores += [s for s,z in zip(live_scores,zombie) if z]
        dead_k = len(dead_samples)
        # remove zombies from the living
        live_samples = [s for s,z in zip(live_samples,zombie) if not z]
        live_scores = [s for s,z in zip(live_scores,zombie) if not z]
        live_k = len(live_samples)

    scores = dead_scores + live_scores
    samples = dead_samples + live_samples
    idx = np.argmin(np.array(scores))
    answer = unroll(samples[idx])

    return answer

def unroll(l):
    x = []

    if type(l) == int:
        return [l]
    for i in l:
        if type(i) == list:
            for v in i :
                x.append(v)
        else: x.append(i)

    return x

