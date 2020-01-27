for k in range(1, k_fold + 1):

        k_model = model()
        data_gen.set_k(k)
        train_loss = 0.0
        train_csi = 0.0
        train_count = 0
        i_batch = 0

        for i in tqdm(range(1, max_iterations)):

            for b, bs in range(data_gen.n_train_batch()):

            for b_val, bs_val in range(data_gen.n_val_batch()):
              print()

      #Test image duay