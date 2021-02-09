import torch
from model import *
import numpy as np
from DataUtils import *
from copy import deepcopy
import matplotlib.pyplot as plt

class Saliency(object):
    """ Abstract class for saliency """

    def __init__(self, model):
        self.model = model
        self.model.eval()


class SaliencyMap(Saliency):
    """Vanilla Saliency to visualize plain gradient information"""

    def __init__(self, model):
        super(SaliencyMap, self).__init__(model)

    def generate_saliency(self, input, target):
        input.requires_grad = True
        input.x.requires_grad = True

        self.model.zero_grad()

        output = self.model((input, device))

        grad_outputs = torch.zeros_like(output)

        grad_outputs[:, target] = 1
        self.model.interpret_embedded_target_gradients.retain_grad()
        self.model.interpret_drug_gradients.retain_grad()
        output.backward(gradient=grad_outputs, retain_graph=True)

        input.requires_grad = False
        input.x.requires_grad = False

        drug_grads = self.model.interpret_drug_gradients.grad
        target_embeded_grads = self.model.interpret_embedded_target_gradients.grad

        return drug_grads, target_embeded_grads


class GuidedBackProp(Saliency):
    """Class for computing guided saliency"""

    def __init__(self, model):
        super(GuidedBackProp, self).__init__(model)

    def guided_relu_hook(self, module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    def generate_saliency(self, input, target):
        input.requires_grad = True
        input.x.requires_grad = True
        self.model.zero_grad()

        for module in self.model.modules():
            if type(module) == nn.ReLU:
                module.register_backward_hook(self.guided_relu_hook)

        output = self.model((input, device))

        grad_outputs = torch.zeros_like(output)

        grad_outputs[:, target] = 1
        self.model.interpret_embedded_target_gradients.retain_grad()
        self.model.interpret_drug_gradients.retain_grad()
        output.backward(gradient=grad_outputs, retain_graph=True)

        input.requires_grad = False
        input.x.requires_grad = False

        drug_grads = self.model.interpret_drug_gradients.grad
        target_embeded_grads = self.model.interpret_embedded_target_gradients.grad

        return drug_grads, target_embeded_grads



def criteria(method):
    method_obj = method(model)
    n_changed_drug = 0
    n_changed_target = 0
    sign_drug_dist = []
    sign_target_dist = []
    for i in tp_indices:
        input_obj = test_data[int(i)]
        input_obj = input_obj.to(device)
        yek, do = method_obj.generate_saliency(input_obj, 1)
        # print(yek.shape)
        # print(do.shape)
        do = do.squeeze(0)

        drug_features = torch.max(yek, 1)[0]
        target_features = torch.max(do, 1)[0]

        significant_drug_features = (drug_features >= drug_features.mean() + drug_features.std()).nonzero()
        significant_target_features = (target_features >= target_features.mean() + target_features.std()).nonzero()
        sign_drug_dist.append(len(significant_drug_features.flatten().tolist()))
        sign_target_dist.append(len(significant_target_features.flatten().tolist()))

        tmp = deepcopy(input_obj)

        input_obj.target[0, significant_target_features] = 0
        model.zero_grad()
        out = model((input_obj, device))
        pred_label_target = np.argmax(out.cpu().detach().numpy())
        if pred_label_target == 0:
            n_changed_target += 1


        tmp.x[significant_drug_features, :] = 0
        model.zero_grad()
        out = model((tmp, device))
        pred_label_drug = np.argmax(out.cpu().detach().numpy())
        if pred_label_drug == 0:
            n_changed_drug += 1
    plt.hist(sign_target_dist)
    plt.savefig(method.__name__ + "_target.png")
    plt.close()
    plt.hist(sign_drug_dist)
    plt.savefig(method.__name__ + "_drug.png")
    plt.close()
    print("Mehtod: ", method.__name__)
    print("Number of changed by manipulating protein : ", n_changed_target, " drug: ", n_changed_drug)
    print("Dist of target:", sign_target_dist)
    print("Dist of drug:", sign_drug_dist)


if __name__ == "__main__":
    device = None
    TEST_BATCH_SIZE = 512
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    model = GINConvNet_Classification()
    model_saved_str = 'model_classificationn_upsample.pt'
    model.load_state_dict(torch.load(model_saved_str))
    for param in model.parameters():
        param.requires_grad = False
    dataset = "davis"
    test_data = TestbedDataset(root='data', dataset=dataset + '_test')
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    model.to(device)


    model.eval()
    total_preds = torch.Tensor()
    total_affinities = torch.Tensor()
    with torch.no_grad():
        for data_object in test_loader:
            output = model(data_object.to(device))
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_affinities = torch.cat((total_affinities, data_object.y.view(-1, 1).cpu()), 0)

    total_affinities = total_affinities.numpy().flatten()
    total_preds = total_preds.numpy()

    preds_classification = np.argmax(total_preds, axis=1)
    affinities_classification = np.where(total_affinities > 7, 1, 0)
    tp_indices = ((preds_classification + affinities_classification) == 2).nonzero()[0]
    print("True class numbers: ", np.sum(affinities_classification==1))
    print("True Positive numbers: ", tp_indices.shape[0])

    criteria(SaliencyMap)
    criteria(GuidedBackProp)