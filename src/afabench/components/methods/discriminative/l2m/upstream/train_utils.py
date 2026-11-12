import torch


def predictor_training_step(config, model, sampler, batch, optimizer, scheduler, accelerator, selector_fn, temp=0.1):
    total_loss = 0
    total_points = 0
    num_available_features = config["num_available_features"]
    m = sampler.initialize_mask(batch)
    optimizer.zero_grad()

    done = torch.zeros(batch['xt'].shape[0], batch['xt'].shape[1], 1, dtype=torch.bool, device=batch['xt'].device)
    for _ in range(0, num_available_features + 1):    
        with accelerator.autocast():
            input_batch = sampler.mask_features(batch, mask=m)
            input_batch = {k: v.detach() for k, v in input_batch.items()}
            outs = model(input_batch, reduce_ll=False) # [B, T]

            done_mask = (~done).float()
            if done_mask.sum() > 0:
                total_loss += (outs['loss'] * done_mask).sum()
                total_points += done_mask.sum()
            else:
                total_loss += torch.tensor(0.0, device=outs['loss'].device)

            done = done | (m >= batch['rt']).all(dim=-1, keepdim=True)

            available = (1 - m) * input_batch['rt']
            random_logits = torch.randn_like(input_batch['xt'])
            khot = selector_fn(random_logits, temp=temp, deterministic=True, feature_groups=sampler.feature_groups, available=available)
            m = torch.max(m, khot)

    # Backwards pass for each acquisition step
    accelerator.backward(total_loss / total_points)
    # PATCH(afabench): optional gradient-norm clipping, which upstream's
    # predictor step does not do. Off unless config['max_grad_norm'] is set.
    if config.get("max_grad_norm") is not None:
        accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    total_loss = total_loss.detach().item() / total_points
    return total_loss

def afa_training_step(config, model, sampler, batch, optimizer, scheduler, accelerator, selector_fn, temp):
    # Rollout trajectory sequentially
        
    inner_steps = config['inner_loop_steps']
    num_available_features = config["num_available_features"]

    for _ in range(inner_steps):
        m = sampler.initialize_mask(batch)
        total_loss = 0
        total_actions = 0

        optimizer.zero_grad()
        for _ in range(1, num_available_features + 1):    
            with accelerator.autocast():
                full_input_batch = {k: v.clone().detach() for k, v in batch.items()}
                input_batch = sampler.mask_features(full_input_batch, mask=m)
                
                logits = model.select_action(input_batch, autoreg=True)

                # Only consider available features for selection
                available = (1 - m) * input_batch['rt']
                has_available = (available.sum(dim=-1) > 0).float().unsqueeze(-1)

                # Select action via sampling from gumbel-softmax
                soft = selector_fn(logits, temp=temp, feature_groups=sampler.feature_groups, available=available)
                #khot = selector_fn(logits, temp=temp, deterministic=True, feature_groups=sampler.feature_groups, available=available)

                m_soft = torch.max(m, soft)
                batch_soft = sampler.mask_features(full_input_batch, mask=m_soft)

                outs = model(batch_soft, reduce_ll=False) # [B, T]
                total_loss += (outs['loss'] * has_available).sum()
                total_actions += has_available.sum().item()

                random_logits = torch.randn_like(input_batch['xt'])
                khot = selector_fn(random_logits, temp=temp, deterministic=True, feature_groups=sampler.feature_groups, available=available)
                m = torch.max(m, khot)

                # probs = logits.softmax(dim=-1).clamp(min=1e-10)  # avoid log(0)
                # entropy = -(probs * probs.log()).sum(dim=-1).mean()  # mean entropy per batch and timestep
                # loss = prediction_loss - config['entropy_weight'] * entropy

        accelerator.backward(total_loss / total_actions)
        # PATCH(afabench): optional gradient-norm clipping, which upstream's
        # sequential AFA step does not do. Off unless config['max_grad_norm']
        # is set.
        if config.get("max_grad_norm") is not None:
            accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        # Compute gradient and update parameters
        # grad_norm = 0.0
        # for name, param in model.selector.named_parameters():
        #     if param.grad is not None:
        #         grad_norm += param.grad.data.norm(2).item() ** 2
        # grad_norm = grad_norm ** 0.5
        # print(f"Gradient norm for model.selector: {grad_norm:.4f}")
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

    total_loss = total_loss.detach().item() / total_actions
    return total_loss


def afa_training_step_random(config, model, sampler, batch, optimizer, scheduler, accelerator, selector_fn, temp):
    total_loss = 0
    
    buffer_size = config['buffer_size']
    inner_steps = config['inner_loop_steps']

    for _ in range(inner_steps):
        optimizer.zero_grad()
        for _ in range(buffer_size):

            with accelerator.autocast():
                input_batch = sampler.mask_features(batch)
                m = input_batch['maskt']
                outs = model.select_action(input_batch, autoreg=True)

                # Only consider available features for selection
                available = (1 - m) * input_batch['rt']
                has_available = (available.sum(dim=-1) > 0).float().unsqueeze(-1)

                # Select action via sampling from gumbel-softmax
                soft = selector_fn(outs, temp=temp, feature_groups=sampler.feature_groups, available=available)
                m_soft = torch.max(m, soft)
                batch_soft = sampler.mask_features(batch, mask=m_soft)

                outs = model(batch_soft, reduce_ll=False, autoreg=True) # [B, T] 
                prediction_loss = (outs['loss'] * has_available).sum() / (has_available.sum() + 1e-8)

                probs = soft.clamp(min=1e-10)  # avoid log(0)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()  # mean entropy per batch and timestep
                loss = prediction_loss - config['entropy_weight'] * entropy
                total_loss += (prediction_loss.detach().item() / buffer_size) / inner_steps
            
            # Accumulate gradient for each buffer sample
            accelerator.backward(loss)

        # Compute gradient and update parameters
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

    return total_loss