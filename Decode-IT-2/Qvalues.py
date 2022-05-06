import torch


class Qvalues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(model, states, actions):
        return model(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def mask_locations(next_states):
        flat = next_states.flatten(start_dim=2)
        maxes = flat.max(dim=2)
        maxes = maxes.values[:, 3].eq(0).type(torch.bool)
        non_final_state_locations = (maxes == False)
        return non_final_state_locations

    @staticmethod
    def masked_states(next_states):
        masks = Qvalues.mask_locations(next_states)
        non_final = next_states[masks]
        return non_final, masks

    @staticmethod
    def get_next(target_net, next_states):
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(Qvalues.device)
        masked_states, masks = Qvalues.masked_states(next_states)
        values[masks] = target_net(masked_states).max(dim=1)[0].detach()
        return values
