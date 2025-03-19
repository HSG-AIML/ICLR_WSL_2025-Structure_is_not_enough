import unittest

import numpy as np
import torch
import torch.nn as nn

from model_distances import *

# Define two sample models
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TestModelDistances(unittest.TestCase):
    def setUp(self):
        self.rdm_testset = torch.utils.data.TensorDataset(torch.rand(1024, 10), torch.randint(0, 9, (1024,)))

    def test_flatten_weights(self):
        model1 = Model()
        model1.fc1.weight.data.fill_(1.0)
        model1.fc1.bias.data.fill_(1.0)
        model1.fc2.weight.data.fill_(1.0)
        model1.fc2.bias.data.fill_(1.0)

        v1 = np.ones((10*5+5+5*2+2))
        w = flatten_weights(model1)

        self.assertEqual(len(v1), len(w))
        self.assertTrue(np.allclose(v1, w),
            f"Expected {v1}, got {w}"
        )

    def test_same_model_has_l2_distance_zero(self):
        model1 = Model()
        distance = distance_matrix([model1, model1], metric='l2')
        self.assertTrue(np.isclose(np.linalg.norm(distance), 0.0),
            f"Expected a distance matrix with norm 0.0, got {np.linalg.norm(distance)}"
        )

    def test_different_model_has_l2_distance_correct(self):
        model1 = Model()
        model1.fc1.weight.data.fill_(0.0)
        model1.fc1.bias.data.fill_(0.0)
        model1.fc2.weight.data.fill_(0.0)
        model1.fc2.bias.data.fill_(0.0)
        v1 = np.zeros((10*5+5+5*2+2))


        model2 = Model()
        model2.fc1.weight.data.fill_(1.0)
        model2.fc1.bias.data.fill_(0.0)
        model2.fc2.weight.data.fill_(0.0)
        model2.fc2.bias.data.fill_(0.0)
        v2 = np.concatenate((np.ones(10*5), np.zeros(5+5*2+2)))

        distance = distance_matrix([model1, model2], metric='l2')

        self.assertTrue(np.isclose(distance[0][0], 0),
            f"Expected {0} at index (0, 0), got {distance}"
        )
        self.assertTrue(np.isclose(distance[0][1], np.linalg.norm(v2-v1, ord=2)),
            f"Expected {np.linalg.norm(v2-v1, ord=2)}, got {distance}"
        )

        model1.fc2.weight.data.fill_(1.0)
        v1[10*5+5:10*5+5+5*2] = 1.0

        distance = distance_matrix([model1, model2], metric='l2')

        self.assertTrue(np.isclose(distance[0][0], 0),
            f"Expected {0} at index (0, 0), got {distance}"
        )
        self.assertTrue(np.isclose(distance[0][1], np.linalg.norm(v2-v1, ord=2)),
            f"Expected {np.linalg.norm(v2-v1, ord=2)}, got {distance}"
        )

    def test_same_model_has_l2_preds_distance_zero(self):
        model1 = Model()

        distance = distance_matrix([model1, model1], 
                                   metric='l2_preds', 
                                   testset='random', 
                                   ds_test=self.rdm_testset
        )

        self.assertTrue(np.isclose(np.linalg.norm(distance), 0.0),
            f"Expected a distance matrix with norm 0.0, got {np.linalg.norm(distance)}"
        )

    def test_different_model_has_l2_preds_distance_nonzero(self):
        model1 = Model()
        model2 = Model()

        distance = distance_matrix([model1, model2], 
                                   metric='l2_preds', 
                                   testset='random', 
                                   ds_test=self.rdm_testset
        )

        self.assertTrue(np.isclose(distance[0][0], 0),
            f"Expected {0} at index (0, 0), got {distance[0][0]}"
        )
        self.assertTrue(np.linalg.norm(distance) > 0.0,
            f"Expected a distance matrix with norm > 0.0, got {np.linalg.norm(distance)}"
        )
        self.assertTrue(np.isfinite(np.linalg.norm(distance)),
            f"Expected a finite distance matrix, got {np.linalg.norm(distance)}"
        )

    def test_same_model_has_kl_divergence_zero(self):
        model1 = Model().to(DEVICE)
        X_test = torch.rand(128, 10).to(DEVICE)
        y_test = compute_predictions(model1, X_test, softmax_flag=True)[1:].reshape(128, -1)

        kl = kl_divergence(y_test, y_test)

        self.assertTrue(np.isclose(kl, 0.0),
            f"Expected 0.0, got {kl}"
        )

    def test_different_model_has_kl_divergence_nonzero(self):
        X_test = torch.rand(128, 10).to(DEVICE)

        model1 = Model().to(DEVICE)
        y_test1 = compute_predictions(model1, X_test, softmax_flag=True)[1:].reshape(128, -1)

        model2 = Model().to(DEVICE)
        y_test2 = compute_predictions(model2, X_test, softmax_flag=True)[1:].reshape(128, -1)

        kl = kl_divergence(y_test1, y_test2)

        self.assertTrue(kl > 0.0,
            f"Expected > 0.0, got {kl}"
        )
        self.assertTrue(np.isfinite(kl),
            f"Expected finite value, got {kl}"
        )

    def test_same_model_has_js_distance_zero(self):
        model1 = Model().to(DEVICE)
        X_test = torch.rand(128, 10).to(DEVICE)
        y_test = compute_predictions(model1, X_test, softmax_flag=True)

        js = js_distance(y_test, y_test)

        self.assertTrue(np.isclose(js, 0.0),
            f"Expected 0.0, got {js}"
        )

    def test_different_model_has_js_distance_nonzero(self):
        X_test = torch.rand(128, 10).to(DEVICE)

        model1 = Model().to(DEVICE)
        y_test1 = compute_predictions(model1, X_test, softmax_flag=True)

        model2 = Model().to(DEVICE)
        y_test2 = compute_predictions(model2, X_test, softmax_flag=True)

        js = js_distance(y_test1, y_test2)

        self.assertTrue(js > 0.0,
            f"Expected > 0.0, got {js}"
        )
        self.assertTrue(np.isfinite(js),
            f"Expected finite value, got {js}"
        )

        js_mat = distance_matrix([model1, model2],
                                  metric='js',
                                  testset='data', 
                                  ds_test=torch.utils.data.TensorDataset(X_test, torch.randint(0, 9, (128,)))
        )

        self.assertTrue(np.isclose(js_mat[0][1], js),
            f"Expected {js} at index (0, 1), got {js_mat[0][1]}"
        )

    def test_js_distance_symmetric(self):
        X_test = torch.rand(128, 10).to(DEVICE)

        model1 = Model().to(DEVICE)
        y_test1 = compute_predictions(model1, X_test, softmax_flag=True)

        model2 = Model().to(DEVICE)
        y_test2 = compute_predictions(model2, X_test, softmax_flag=True)

        js1 = js_distance(y_test1, y_test2)
        js2 = js_distance(y_test2, y_test1)

        self.assertTrue(np.isclose(js1, js2),
            f"Expected {js1} == {js2}"
        )

    def test_same_model_has_cka_distance_zero(self):
        model1 = Model().to(DEVICE)
        X_test = torch.rand(128, 10).to(DEVICE)
        y_test = compute_predictions(model1, X_test, centre_flag=True)

        ckad = cka_distance(y_test, y_test)
        self.assertTrue(np.isclose(ckad, 0.0, atol=1e-6),
            f"Expected 0.0, got {ckad}"
        )

        ckad = distance_matrix([model1, model1],
                               metric='cka',
                               testset='data', 
                               ds_test=self.rdm_testset
        )
        self.assertTrue(np.isclose(ckad[0][1], 0.0, atol=1e-6) and np.isclose(ckad[1][0], 0.0, atol=1e-6),
            f"Expected a distance matrix with 0.0 in position [0,1], got {ckad[0][1]}"
        )

    def test_different_model_has_cka_distance_nonzero(self):
        X_test = torch.rand(128, 10).to(DEVICE)

        model1 = Model().to(DEVICE)
        y_test1 = compute_predictions(model1, X_test, centre_flag=True)

        model2 = Model().to(DEVICE)
        y_test2 = compute_predictions(model2, X_test, centre_flag=True)

        ckad = cka_distance(y_test1, y_test2)
        self.assertTrue(ckad > 0.0,
            f"Expected > 0.0, got {ckad}"
        )
        self.assertTrue(ckad <= 1.0,
            f"Expected <= 1.0, got {ckad}"
        )
        ckad_mat = distance_matrix([model1, model2],
                                   metric='cka',
                                   testset='data', 
                                   ds_test=torch.utils.data.TensorDataset(X_test, torch.randint(0, 9, (128,)))
        )
        self.assertTrue(np.isclose(ckad_mat[0][1], ckad),
            f"Expected {ckad} at index (0, 1), got {ckad_mat[0][1]}"                
        )

    def test_cka_distance_symmetric(self):
        X_test = torch.rand(128, 10).to(DEVICE)

        model1 = Model().to(DEVICE)
        y_test1 = compute_predictions(model1, X_test, centre_flag=True)

        model2 = Model().to(DEVICE)
        y_test2 = compute_predictions(model2, X_test, centre_flag=True)

        ckad1 = cka_distance(y_test1, y_test2)
        ckad2 = cka_distance(y_test2, y_test1)

        self.assertTrue(np.isclose(ckad1, ckad2),
            f"Expected {ckad1} == {ckad2}"
        )

    def test_same_model_has_l2_worst_distance_zero(self):
        model1 = Model().to(DEVICE)
        X_test = torch.rand(128, 10).to(DEVICE)
        y_test = compute_predictions(model1, X_test, softmax_flag=True)

        l2w = l2_worst_distance(y_test, y_test)

        self.assertTrue(np.isclose(l2w, 0.0),
            f"Expected 0.0, got {l2w}"
        )

        l2w = distance_matrix([model1, model1],
                              metric='l2_worst',
                              testset='data', 
                              ds_test=self.rdm_testset
        )

        self.assertTrue(np.isclose(l2w[0][1], 0.0) and np.isclose(l2w[1][0], 0.0),
            f"Expected a distance matrix with 0.0 in position [0,1], got {l2w[0][1]}"
        )

    def test_different_model_has_l2_worst_distance_nonzero(self):
        X_test = torch.rand(128, 10).to(DEVICE)

        model1 = Model().to(DEVICE)
        y_test1 = compute_predictions(model1, X_test, softmax_flag=True)

        model2 = Model().to(DEVICE)
        y_test2 = compute_predictions(model2, X_test, softmax_flag=True)

        l2w = l2_worst_distance(y_test1, y_test2)

        self.assertTrue(l2w > 0.0,
            f"Expected > 0.0, got {l2w}"
        )
        self.assertTrue(np.isfinite(l2w),
            f"Expected finite value, got {l2w}"
        )

        l2w_mat = distance_matrix([model1, model2],
                                  metric='l2_worst',
                                  testset='data', 
                                  ds_test=torch.utils.data.TensorDataset(X_test, torch.randint(0, 9, (128,)))
        )

        self.assertTrue(np.isclose(l2w_mat[0][1], l2w),
            f"Expected {l2w} at index (0, 1), got {l2w_mat[0][1]}"
        )