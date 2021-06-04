image = """<img src="https://monebo.com/wp-content/uploads/2019/06/output_76TFOb.gif" width="100"></img>"""

dataset = """ The original dataset for "ECG5000" is a 20-hour long ECG downloaded from Physionet. 
        The name is BIDMC Congestive Heart Failure Database(chfdb) and it is record "chf07". 
        It was originally published in "Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23)".
        The data was pre-processed in two steps: (1) extract each heartbeat, (2) make each heartbeat equal length using interpolation. 
        This dataset was originally used in paper "A general framework for never-ending learning from time series streams", DAMI 29(6). After that, 5,000 heartbeats were randomly selected. 
        The patient has severe congestive heart failure and the class values were obtained by automated annotation. You can download the data here: http://timeseriesclassification.com/Downloads/ECG5000.zip """

model = """This model uses an autoencoder architecture. An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. 
        They work by compressing the input into a latent-space representation, and then reconstructing the output from this representation. 
        Autoencoders consist of two components - an encoder and a decoder. The encoder learns a mapping from an image to a lower-dimensional latent space, and the decoder learns a mapping from the latent space back to the original image. 
        In this way, autoencoders are trained in an unsupervised manner by minimizing the error between the original image and the reconstruction.
        """

normal_explain = """This graph shows an instance of Normal ECG Data compared to the model's decoded output - the reconstruction created by the model.
            As seen in the graph, the decoded output fits the normal test data very closely. The reconstruction error, which is the difference between the test data (blue) and the decoded output (red), 
            is very low. This is because the model was only trained on normal data.
            """

anomaly_explain = """This graph shows an instance of Anomaly ECG Data compared to the model's decoded output - the reconstruction created by the model.
            As seen in the graph, the decoded output does not fit the anomaly test data closely. The reconstruction error, which is the difference between the test data (blue) and the decoded output (red), 
            is high. This is because the model was only trained on normal data, not on anomaly data. Thus, it is difficult for the autoencoder to recontruct this type of ECG data.
            """

predictions_text = """ We can use the information from the graphs above to detect whether an ECG shows normal data or anomaly data. This can be done using the reconstruction error.
    For this, the Mean Absolute Error between the reconstruction and the test data (normal or anomaly) can be calculated. The mean and standard deviation of this error can be then used 
    to identify a threshold value which can be used to differentiate between normal and anomaly ECG data. When calculated using the error between the normal data and reconstruction, 
    any values below the threshold would be classified as normal data and any values above the threshold was be classified as anomaly data. This makes sense because the recontruction closely fits the normal data as 
    it was trained on it, and thus the loss/error should be lower.
    """

threshold = """The graph to the right shows the clear separation betweent the reconstruction loss 
        for the normal (green) data and the anomaly (purple) data by the threshold (orange line). Some of the normal and anomaly data crosses past the threshold value - 
        this would be classified as false positives. For the most part, the threshold does a good job of separating these data."""

evaluation = """Below, you can see how the model performs on data it has not seen before - both normal and abnormal ECGs. From the data, 
    it seems to have a high accuracy score on both the normal and anomaly data."""