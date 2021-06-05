image = """<img src="https://monebo.com/wp-content/uploads/2019/06/output_76TFOb.gif" width="100"></img>"""

dataset = """The original dataset for "ECG5000" is a 20-hour long ECG downloaded from Physionet. 
        It was originally published in "Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23)".
        The data was pre-processed in two steps: (1) extract each heartbeat, (2) make each heartbeat equal length using interpolation. 
        This dataset was originally used in paper "A general framework for never-ending learning from time series streams", DAMI 29(6). After that, 5,000 heartbeats were randomly selected. 
        The patient has severe congestive heart failure and the class values were obtained by automated annotation. You can download the data here: http://timeseriesclassification.com/Downloads/ECG5000.zip """

model = """This model uses an autoencoder architecture. An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. 
        It works by compressing input data to create a latent-space representation, and then reconstructing the data from this representation to create the output. 
        An autoencoder consists of two components - an encoder and a decoder. The encoder first learns a mapping from the data to a lower-dimensional latent space, and the decoder then learns a mapping from the latent space back to the original higher-dimensional space. 
        **It is important to note that the model is trained only on normal data.** This makes it learn in an unsupervised manner by minimizing the error between the original data and the reconstructed data.
        """

normal_explain = """This graph shows an instance of Normal ECG Data compared to the model's decoded output - the reconstruction created by the model.
            As seen in the graph, the reconstructed data fits the normal test data very closely. The reconstruction error, which is the difference between the test data (blue) and the reconstructed data (red), 
            is very low. This is because the model was trained on normal data and so it has learned to generalize it very well.
            """

anomaly_explain = """This graph shows an instance of Anomaly ECG Data compared to the model's decoded output - the reconstruction created by the model.
            As seen in the graph, the reconstructed data **does not** fit the anomaly test data closely. The reconstruction error, which is the difference between the test data (blue) and the reconstructed data (red), 
            is high. This is because the model was trained on normal, and not anomaly data. Thus, it is difficult for the autoencoder to generalize and reconstruct it.
            """

predictions_text = """The graphs above help to display the differences between how the model learns normal versus anomaly ECG data. The key component is the resuling reconstruction error.
    To calculate this, the Mean Absolute Error between the reconstruction and test data (normal or anomaly) is needed. The mean and standard deviation of this error can be then used 
    to identify a **threshold value** which can be used to differentiate between normal and anomaly data. In simple terms, "if the difference between the data in the original and the reconstruction is greater than the 
    threshold value, then it is anomalous. On the other hand, if it's less than the threshold, it is non-anomalous". This makes sense because the recontruction closely fits the normal data on which it was trained, and thus the loss/error should be low.
    """

threshold = """The graph to the right shows the clear separation betweent the reconstruction loss 
        for the normal (green) data and the anomaly (purple) data by the threshold (orange line). Some of the normal and anomaly data crosses past the threshold value - 
        this would be classified as false positives/negatives. For the most part, the threshold does a good job of separating the data."""

evaluation = """Below, you can see how the model performs on data it has not seen before - both normal and abnormal ECGs. From the table, 
    it seems to have a high accuracy score on both the normal and anomaly data."""