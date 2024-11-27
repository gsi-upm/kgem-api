from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

import torch

import matplotlib.pyplot as plt

from tqdm import tqdm

def train_embedding_model(graph_name, dataset:str = "", embedding_model:str="TransE",embedding_dim=3,
                          loss="softplus", random_seed=1, device="gpu", num_epochs=15, batch_size=50, lr=1.0e-3, 
                          triples_from_dataset=True, triples_from_path=False, split_size=(0.8,0.1), triples_path="", 
                          models_route="models", plot_results=False):

    """
    Train an embedding model using knowledge graph data.

    Args:
        graph_name (str): A verbous name for the graph for saving the model.
        dataset (str, optional): Name of the Pykeen dataset to be used for training. Default is an empty string.
        embedding_model (str, optional): Name of the embedding model (e.g., 'TransE', 'ComplEx'). Default is 'TransE'.
        embedding_dim (int, optional): Dimension of the embedding vectors. Default is 3.
        loss (str, optional): Loss function to use during training. Default is 'softplus'.
        random_seed (int, optional): Random seed for reproducibility. Default is 1.
        device (str, optional): Device for computation ('cpu' or 'gpu'). Default is 'gpu'.
        num_epochs (int, optional): Number of training epochs. Default is 15.
        batch_size (int, optional): Size of training batches. Default is 50.
        lr (float, optional): Learning rate for optimization. Default is 1.0e-3.
        triples_from_dataset (bool, optional): Whether to load triples from a predefined Pykeen dataset. Default is True.
        triples_from_path (bool, optional): Whether to load triples from a file path. Default is False.
        split_size (tuple, optional): Proportions for splitting data into training, validation, and test sets if "triples_from_path" is True. Default is (0.8, 0.1).
        triples_path (str, optional): Path to the triples file, if `triples_from_path` is True. Default is an empty string.
        models_route (str, optional): Directory to save trained models. Default is 'models'.
        plot_results (bool, optional): Whether to plot and save loss graphs. Default is False.

    Returns:
        pipeline.PipelineResult: The result of the training pipeline, including trained model and metrics.

    Raises:
        ValueError: If both `triples_from_path` and `triples_from_dataset` are False or mutually exclusive conditions are invalid.
        FileNotFoundError: If `triples_from_path` is True but `triples_path` is not valid.

    """
    if not (triples_from_path or triples_from_dataset):
        raise ValueError("Either `triples_from_path` or `triples_from_dataset` must be True.")
    
    if triples_from_path and not triples_path:
        raise FileNotFoundError("`triples_path` must be provided when `triples_from_path` is True.")

    print("Running on GPU?", torch.cuda.is_available())

    if triples_from_path:
        factory = TriplesFactory.from_path(triples_path)
        training_factory, testing_factory, validation_factory = factory.split(split_size)

        pipeline_result = pipeline(
            loss=loss,
            training=training_factory,
            validation=validation_factory,
            testing=testing_factory,
            model=embedding_model,
            model_kwargs=dict(embedding_dim=embedding_dim),
            random_seed=random_seed,
            device=device,
            training_kwargs=dict(num_epochs=num_epochs, use_tqdm_batch=True, batch_size=batch_size),
            optimizer_kwargs=dict(lr=lr),
            stopper="early",
            stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
        )

    elif triples_from_dataset:
        pipeline_result = pipeline(
            dataset=dataset,
            loss=loss,
            model=embedding_model,
            model_kwargs=dict(embedding_dim=embedding_dim),
            random_seed=random_seed,
            device=device,
            training_kwargs=dict(num_epochs=num_epochs, use_tqdm_batch=True, batch_size=batch_size),
            optimizer_kwargs=dict(lr=lr),
            stopper="early",
            stopper_kwargs=dict(frequency=5, patience=3, relative_delta=0.002),
        )

    save_path = f"{models_route}/{graph_name.lower()}"
    pipeline_result.save_to_directory(save_path)

    if plot_results:
        pipeline_result.plot_losses()
        plt.savefig(f"{save_path}/losses.jpg")

    return pipeline_result




if __name__=="__main__":

    models_route="models" # directory where you store the trained kge models

    train_embedding_model(graph_name="nations",
                          dataset="nations",
                          embedding_model="transe",
                          embedding_dim=5,
                          random_seed=1234,
                          models_route=models_route,
                          triples_from_dataset=True)
