 print("Using Geometric Backend")
        if model_spec == 'gs-mean':
            model = geo.GraphSAGE(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == "gcn":
            model = geo.GCN(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == "gat":
            print("Warning, GAT doesn't respect n_layers")
            heads = [8, args.gat_out_heads]  # Fixed head config
            n_hidden_per_head = int(n_hidden / heads[0])
            model = geo.GAT(in_feats, n_hidden_per_head, n_classes, F.relu, args.dropout, 0.6, heads).to(device)
        elif model_spec == "mlp":
            model = geo.MLP(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif model_spec == 'jknet-sageconv':
            # Geometric JKNEt with SAGECOnv
            model = JKNet(tg.nn.SAGEConv, in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                    mode="cat", conv_kwargs={"normalize": False}, backend="geometric").to(device)
        elif model_spec == 'jknet-graphconv':
            model = JKNet(tg.nn.GraphConv, in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout,
                    mode="cat", conv_kwargs={"aggr": "mean"}, backend="geometric").to(device)
        elif model_spec == "sgnet":
            model = geo.SGNet(in_channels=in_feats, out_channels=n_classes, K=n_layers, cached=True).to(device)
        else:
            raise NotImplementedError(f"Unknown model spec 'f{model_spec} for backend {backend}")