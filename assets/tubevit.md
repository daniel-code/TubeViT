Rethinking Video ViTs: Sparse Video Tubes for Joint Image and Video Learning
|     |     | AJPiergiovanni |     |     |     | WeichengKuo |     |     | AneliaAngelova |     |     |     |
| --- | --- | -------------- | --- | --- | --- | ----------- | --- | --- | -------------- | --- | --- | --- |
Abstract
Image-Only Stream
2D Patches
| 2202 ceD 6  ]VC.sc[  1v92230.2122:viXra     |     |     |     |     |     |     |     |     |                              |     | …   |                   |
| ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | ----------------- |
| WepresentasimpleapproachwhichcanturnaViTen- |     |     |     |     |     |     |     |     | Video Stream                 |     |     |                   |
|                                             |     |     |     |     |     |     |     |     | 2D Patches from a few frames |     |     | ViT Encoder Class |
|                                             |     |     |     |     |     |     |     |     |                              |     | …   | output            |
coder into an efficient video model, which can seamlessly 3D Sparse Tubes
| work with | both image | and | video | inputs. By | sparsely | sam- |     |     |     |     | …   |     |
| --------- | ---------- | --- | ----- | ---------- | -------- | ---- | --- | --- | --- | --- | --- | --- |
plingtheinputs,themodelisabletodotrainingandinfer-
|                     |     |                                |     |     |     |     | Figure1. | TubeViT:WithSparseVideoTubes,VisionTransform- |     |     |     |     |
| ------------------- | --- | ------------------------------ | --- | --- | --- | --- | -------- | --------------------------------------------- | --- | --- | --- | --- |
| encefrombothinputs. |     | Themodeliseasilyscalableandcan |     |     |     |     |          |                                               |     |     |     |     |
ers(ViTs)usebothimageandvideoinputs,providiTunbe gPatcheasnefficient
| be adapted | to large-scale |     | pre-trained | ViTs | without | requir- |     |     |     |     |     |     |
| ---------- | -------------- | --- | ----------- | ---- | ------- | ------- | --- | --- | --- | --- | --- | --- |
videobackboneandmoreaccurateperformance.
| ing full finetuning.       |     | The model | achieves | SOTA | results | and |     |     |     |     |     |     |
| -------------------------- | --- | --------- | -------- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- |
| thecodewillbeopen-sourced. |     |           |          |      |         |     |     |     |     | …   |     |     |
+ Position Embedding (Fixed Cosine/Sine)
features.However,thisresamplingcanstillbeexpensivefor
longvideos,and,inthecaseofFlamingo,ittreatsvideosas
1.Introduction
|     |     |     |     |     |     |     | individualframessampled |     |     | at1FPS,whichlimits |     | thetem- |
| --- | --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ------------------ | --- | ------- |
Visual Transformers (ViT) [10] have been an ubiqui- poral information. Such low FPS sampling and per-frame
modelingwouldoftenbeinsufficientfordatasetswhichrely
tousbackboneforvisualrepresentationlearning,leadingto
many advances in image understanding [45,55,67], mul- on motion and temporal understanding, e.g., Something-
Something[18],orforrecognizingquickandshortactions.
| timodal tasks | [1, | 2, 63, 66] | and | self-supervised |     | learning |     |     |     |     |     |     |
| ------------- | --- | ---------- | --- | --------------- | --- | -------- | --- | --- | --- | --- | --- | --- |
[5,15,47], etc. However, adaptations to video are both On the other hand, using one of the above-mentioned ap-
challenging and computationally intensive, so video ver- proacheswithdenseframesiscomputationallyinfeasible.
sionshavebeenbeenspeciallydesignedtohandlethelarger Toaddresstheselimitations,weproposeasimplebutef-
numberofframes,forexample,ViViT[3],MultiView[62], fective model, named TubeViT, to utilize a standard ViT
|     |     |     |     |     |     |     | model seamlessly |     | for | both image | and videos. | We intro- |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | --- | ---------- | ----------- | --------- |
TimeSFormer[6]andothers[13].
Video understanding is an essential computer vision duce Sparse Video Tubes, a lightweight approach for joint
task, and a large number of successful video architectures image and video learning. Our method works by sparsely
have been developed [8,14,16,28,39,48,56,61]. Previ- samplingvarioussized3Dspace-timetubesfromthevideo
ousvideo3DCNNs[8,48]weredesignedtohandlevideos to generate learnable tokens, which are used by the vision
|     |     |     |     |     |     |     | transformer(Figure1). |     |     | Withsparsevideotubes,themodel |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | --- | ----------------------------- | --- | --- |
bylearningspatio-temporalinformation;theyoftenborrow
from mechanisms for learning on images, for example [8] is easily applicable to either input, and can better leverage
usepre-trainedimageCNNweightsbyinflatingthekernels either or both sources of data for training and fine-tuning.
|                 |      |         |     |               |         |     | The sparse | video | tubes | naturally | handle raw | video signals |
| --------------- | ---- | ------- | --- | ------------- | ------- | --- | ---------- | ----- | ----- | --------- | ---------- | ------------- |
| to 3D. However, | once | adapted | to  | videos, these | kernels | are |            |       |       |           |            |               |
nolongerapplicabletoimages. andimagesignalswhichiscrucialtounderstandingactions
andotherspatio-temporalinformationinvideos.
Furthermore,mostpreviousworkstreatimageandvideo
asentirelydifferentinputs,providingindependentmethods Video models are also expensive to train, and previous
|            |           |         |       |           |         |     | works have | studied | ways | to  | leverage already | trained mod- |
| ---------- | --------- | ------- | ----- | --------- | ------- | --- | ---------- | ------- | ---- | --- | ---------------- | ------------ |
| for either | videos or | images, | since | designing | a model | ca- |            |         |      |     |                  |              |
pable of handling both is challenging. At the same time, els, such as using frozen ones [27] or adapting them to
image and video inputs are inherently related and a single videos[31]. Weexpandontheseideas,andusetheSparse
|     |     |     |     |     |     |     | Video Tubes | to  | adapt | much | larger ViT models | to videos |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ----- | ---- | ----------------- | --------- |
visualbackboneshouldbeabletohandleeitherorbothin-
puts. Previous methods for co-training image and video withlightweighttraining(Sec. 3.6). Thuswecreatepower-
[4,25,53,68]adaptthearchitecturestodosowithsignificant fullargevideomodelswithlessresources.
portionsofthenetworkdesignedforeachinput.Workssuch We evaluate the approach across many standard video
as Perceiver [19] and Flamingo [2] address this by resam- datasets: Kinetics-400, Kinetics-600, Kinetics-700, and
pling the input and compressing it into a fixed number of SomethingSomething V2, outperforming the state-of-the-

| art (SOTA). | Our | methods  | are | trained  | from | scratch    | or on |     |     | Class |     |     |
| ----------- | --- | -------- | --- | -------- | ---- | ---------- | ----- | --- | --- | ----- | --- | --- |
| ImageNet-1k | and | Kinetics |     | datasets | and  | outperform | even  |     |     |       |     |     |
methods additionally pre-trained from very large datasets Attention Pooling + FC
| (e.g.,JFT[46]). |              | Ourworkalsooutperformsmodelstarget- |      |           |       |        |       |     |     |     |     |     |
| --------------- | ------------ | ----------------------------------- | ---- | --------- | ----- | ------ | ----- | --- | --- | --- | --- | --- |
| ing video       | pretraining, |                                     | such | as recent | video | Masked | Auto- |     |     |     |     |     |
Encoder(MAE)works[15,47].
ViT Encoder
Ourkeyfindingsarethatbyusingthesparsevideotubes,
weareabletobettersharetheweightslearnedforbothim-
| ages and       | videos.    | This       | is in     | contrast              | to         | prior  | works that |     |              |     |     |            |
| -------------- | ---------- | ---------- | --------- | --------------------- | ---------- | ------ | ---------- | --- | ------------ | --- | --- | ---------- |
| either inflate | kernels    |            | or add    | new temporal-specific |            |        | layers.    |     |              |     |     |            |
| Further,       | due to     | the sparse | sampling, |                       | the        | number | of tokens  |     |              |     |     |            |
| remains        | low, which | we         | also      | find is               | important, | both   | for re-    |     |              |     |     |            |
|                |            |            |           |                       |            |        |            |     | Tube Patches |     |     | 2D Patches |
ducingFLOPsandimprovingperformance.
+ Position Embedding (Fixed Cosine/Sine)
Ourcontributionisconstructionofsparsevideotubes,
obtainedbysparselysamplingvideoswithvarioussized3D
| space-time      | tubes.   | With       | that     | we accomplish  |        | the       | following: |     |     |     |     |     |
| --------------- | -------- | ---------- | -------- | -------------- | ------ | --------- | ---------- | --- | --- | --- | --- | --- |
| (1) a universal |          | visual     | backbone | which          | easily | adapts    | a ViT      |     |     |     |     |     |
| architecture    | to       | videos;    | (2)      | joint image    |        | and video | under-     |     |     |     |     |     |
| standing        | which    | seamlessly |          | uses either    | input; | (3)       | an easy-   |     |     |     |     |     |
| to-scale        | approach | for        | video    | understanding, |        | which     | can also   |     |     |     |     |     |
leveragealreadytrained(large)ViTmodels.
2.Relatedwork
…
| Video | understanding |     | is an | important | topic | in  | computer |     |     |     |     |     |
| ----- | ------------- | --- | ----- | --------- | ----- | --- | -------- | --- | --- | --- | --- | --- |
vision.Earlyworkshand-designedtrajectoryfeaturestoun-
|                            |     |     |     |                        |     |     |     | Figure2. | Illustrationoftheapproach. |     | Weusetubesofdifferent |     |
| -------------------------- | --- | --- | --- | ---------------------- | --- | --- | --- | -------- | -------------------------- | --- | --------------------- | --- |
| derstandmotionandtime[52]. |     |     |     | Withthesuccessofneural |     |     |     |          |                            |     |                       |     |
networks,manydifferentapproacheshavebeendeveloped, shapestosparselysamplethevideo. Theseareconcatenatedto-
getherandusedasinputtoatransformermodel.
suchastwo-streamCNNstakingimageframesplusoptical
| flow for     | motion | information |      | as input     | [43], | finding | a clear  |     |     |     |     |     |
| ------------ | ------ | ----------- | ---- | ------------ | ----- | ------- | -------- | --- | --- | --- | --- | --- |
| benefit from | adding | the         | flow | information. |       | Works   | studying |     |     |     |     |     |
beroftokensinvideotransformermodels[32,38,54].How-
3DCNNsfoundthelearningoftemporalkernelstobeim-
ever,alltheseworksstilluseaninitialdensesamplingofthe
portant[8,34,48,50],butalsorequiredmuchmoredatain
video,thensomeheuristicstoreducethenumberofinputs.
| ordertobe   | effective[8]. |      | Manyoftheexistingvideo |     |           |         | CNN    |               |         |                 |     |                  |
| ----------- | ------------- | ---- | ---------------------- | --- | --------- | ------- | ------ | ------------- | ------- | --------------- | --- | ---------------- |
|             |               |      |                        |     |           |         |        | In this work, | we more | sparsely sample | the | input initially, |
| approaches, | have          | been | specialized            |     | to handle | videos, | either |               |         |                 |     |                  |
increasingefficiency.
withflowstreamsor3Dkernelsandthushavenotbeenap-
OtherrecentworkshavestudiedvideoMAEtasksaspre-
plicabletoimages.
With the introduction of transformer models and self- training [15,47], they similarly treat videos as tubes, and
|     |     |     |     |     |     |     |     | studythesparsenessintermsofthemasking, |     |     |     | havingsimi- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------- | --- | --- | --- | ----------- |
attention[51],visiontransformershavebeenveryeffective
|                 |     |        |          |     |     |               |      | larfindingsthatsparsenessisbeneficial. |     |     | However,theyuse |     |
| --------------- | --- | ------ | -------- | --- | --- | ------------- | ---- | -------------------------------------- | --- | --- | --------------- | --- |
| for image-based |     | tasks. | However, | due | to  | the quadratic | cost |                                        |     |     |                 |     |
asingletubeshapeandcreatenon-overlappingpatchesand
ofself-attentionandthedensesampling,theiruseforvideos
havenotbeenstudiedwhenjointtrainingwithimages.
| has required | different |     | elements, | such | as space-time |     | factor- |     |     |     |     |     |
| ------------ | --------- | --- | --------- | ---- | ------------- | --- | ------- | --- | --- | --- | --- | --- |
izedattention[3,6,62]. However,thesevideotransformers This work is also related to approaches which use mul-
|     |     |     |     |     |     |     |     | tiple views | or streams | from the input | data, | e.g., Multi- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---------- | -------------- | ----- | ------------ |
havenotreallybeentestedonlongervideosandaremostly
evaluatedonshortclips. Theabilitytohandlelargernum- ViewTransformers[62], SlowFastNetworks[16]andoth-
ber of input frames and understand long-term actions and ers [35,43], all have found benefits from multiple input
theirrelationshipsisofkeyimportance,butbecomescom- views or streams. MultiView Transformers [62], similarly
putationallyprohibitivewithcurrentmodels. tous, isusingtubesofvaryingshapes. Thekeydifference
|          |       |      |       |      |              |     |          | is the sparse | sampling | we use enables | the | use of a single |
| -------- | ----- | ---- | ----- | ---- | ------------ | --- | -------- | ------------- | -------- | -------------- | --- | --------------- |
| Previous | works | have | found | that | transformers |     | focus on |               |          |                |     |                 |
only a few tokens [30,37] and works have been designed ViT encoder model, rather than multiple smaller, per-view
topoolorreorganizedtokenseffectively[24,29,38]. Many encoders. Thisfurtherunifiestheapproachwithimages.
video works have found that frames contain redundant in- Anotherlineofworkinvideounderstandingisleverag-
formation, and thus propose strategies to sample frames ingimagedatasetsduringpre-training[12,57].Thisisvalu-
[17,60]. Otherworkshavestudiedwaystoreducethenum- ableasimage-onlydatasetsarebetterannotatedandprovide

richer semantic information. One approach is to bootstrap images: a2Dconvolutionwitha16×16kernel. Webuild
the video models from image-pretrained models, often by on the observation that sparseness is effective for videos.
inflating kernels. The model is first pre-trained on image Ratherthanfollowingthepriorworksthatdenselytokenize
data,andthenonlytrainedonvideo. Otherworksproposed the video, we instead use the same 2D kernel, but with a
to co-train image and video jointly [4,19,25,53,57,68]. large temporal stride, for example, applied to every 16th
Theseapproachesadaptthearchitecturestohandlebothin- frame. Thusforaninputvideoclipof32×224×224,this
puts which might be inefficient, e.g., treating an image in- results in only 392 tokens, rather than the 6k in TimeS-
putasavideoof1frames[68]orusingseparatenetworks Formeror1-2kinViViT.
tofirstencodetheinputs[2,19]. However, this sparse spatial sampling might lose infor-
Incontrasttoallthepreviousworks,ourmethodissim- mation,especiallyforquickorshortactions. Thus,wecre-
ple and straightforward. One crucial set of differences is atesparsetubesofdifferentshapes,forexample,a16×4×4
that the tubes are sparsely applied to the raw input, con- tubetoobtaininformationfrommanyframesatlowspatial
sists of different shaped, possibly overlapping tubes, and resolution. These tubes can have any shape, and we ex-
uses a single, shared backbone network, different from all perimentallyexploretheeffectofthese. Importantly,these
previousapproaches([3,15,16,32,38,47,62]). Thisleads tubesalsohavelargestrides,sparselysamplingthevideoin
toboth moreefficientandaccurate models. Secondly, and differentviews. Wealsooptionallyaddanoffsettothestart
moreimportantly,themodelisentirelysharedbetweenthe location, sothatthepatchesdonotalwaysstartat(0,0,0)
| image and video | modalities. | This is an important | distinc- |     |     |     |     |
| --------------- | ----------- | -------------------- | -------- | --- | --- | --- | --- |
andthisallowsareductionintheoverlapbetweenthetubes.
tionasitnotonlyimprovesperformanceforbothtasks,but This is illustrated in Figure 2. Tubes of various sizes are
isalsomoregenerallyapplicabletovisiontasks. alsousedintheMultiViewapproachforvideoclassification
[62],howevertheretheyaredenselysampledandprocessed
3.Method
bymultipletransformers,resultinginamorecomputation-
| 3.1.Preliminaries |     |     |     | allyintensiveapproach. |     |     |     |
| ----------------- | --- | --- | --- | ---------------------- | --- | --- | --- |
Furthermore,incontrasttopriorworks,wealsoallowfor
The standard ViT architecture [10] takes an image and overlapbetweenthetubes. Specifically,wecanrepresenta
converts it into patch embedding, for example, by using a tubeas(T×H×W)forthekernelshape,(T ,H ,W )for
s s s
16 × 16 2D convolutional kernel, with a 16 × 16 stride. thespatio-temporalstrideappliedtothekernel,and(x,y,z)
Thisresultsinasequenceofpatchesastheimagerepresen-
astheoffsetofthestartingpointoftheconvolution.
tation,e.g.,196fora224×224inputimage. Givenavideo With the proposed design, our approach enables seam-
V ∈ RT×H×W×C,priorapproacheseitherusedthesame,
lessfusionoftheimage-andvideo-visualinformation.The
dense 2D patches (e.g., TimeSFormer [6]) or used dense sparsespatialsamplingallowssharingtheimageandframe
3Dkernels,e.g.,2or4×16×16asinViViT[3]. Inboth tokens and the sparse video tubes create a low number of
cases,thisresultsinsignificantlymoretokens,e.g.,T∗196,
video-specifictokens.ThisenablesbettersharingoftheViT
whereT isthenumberofframes.Thesetubesorpatchesare modelbetweenimagesandvideos.
| thenlinearlyprojectedintoanembeddingspace, |     |     | z ∈ Rd. |     |     |     |     |
| ------------------------------------------ | --- | --- | ------- | --- | --- | --- | --- |
i
Thissequenceoftokensisthenprocessedbyatransformer 3.3.Positionalembeddingforsparsevideotubes
encoder,usingstandardcomponents,MSA-themulti-head
|     |     |     |     | A key aspect | of our | approach is the | implementation |
| --- | --- | --- | --- | ------------ | ------ | --------------- | -------------- |
selfattentionandMLP-thestandardtransformerprojection
|                   |       |                       |           | of the positional | embedding. | In language | models, rela- |
| ----------------- | ----- | --------------------- | --------- | ----------------- | ---------- | ----------- | ------------- |
| layer (LN denotes | Layer | Norm). For a sequence | of layers |                   |            |             |               |
tivepositionalembeddingsareacommonandeffectiveap-
| l ∈ [0,1,...L],wecomputetherepresentationyl |            |         | andnext |                 |                    |                    |                  |
| ------------------------------------------- | ---------- | ------- | ------- | --------------- | ------------------ | ------------------ | ---------------- |
|                                             |            |         | i       | proach [51,58]. | However,           | here, the relative | position be-     |
| tokenfeatureszl                             | forallthez | tokens: |         |                 |                    |                    |                  |
|                                             | i          | i       |         |                 |                    |                    |                  |
|                                             |            |         |         | tween two       | tokens has minimal | meaning,           | and no real ref- |
yl =MSA(LN(zl−1))+zl−1 (1) erence to where the patch/tube came from in the original
|     | i                  | i i |     |                                                   |                                   |     |     |
| --- | ------------------ | --- | --- | ------------------------------------------------- | --------------------------------- | --- | --- |
|     |                    |     |     | videoorimage.                                     | TheViTmodel[10]andsimilarlyTimeS- |     |     |
|     | zl =MLP(LN(yl))+yl |     | (2) |                                                   |                                   |     |     |
|     | i                  | i i |     | Former[6]andViViT[3]usedlearnablepositionalembed- |                                   |     |     |
Toreducethecomputationalcost,priorapproachesfac- dingsforthepatches. Here, suchanapproachcanbehard
torize the attention mechanism, to have a spatial and tem- for the model, as these learned embeddings do not neces-
poralattention[3]orusemultipleviewswithsmaller,view sarily reflect where the patches came from in the original
leveltransformers[62]. video,especiallyinthecasewherepatchesoverlap.
|     |     |     |     | Instead, | we use a fixed | sine/cosine embedding. | Impor- |
| --- | --- | --- | --- | -------- | -------------- | ---------------------- | ------ |
3.2.SparseVideoTubes
|     |     |     |     | tantly, we take | into account | the stride, | kernel shape and |
| --- | --- | --- | --- | --------------- | ------------ | ----------- | ---------------- |
Weproposeasimpleandstraightforwardmethodwhich offsets of each tube when applying the positional embed-
isseamlesslyapplicabletobothimagesandvideos.Ourap- dings. This ensures that the positional embedding of each
proachfollowsthestandardViTtokenizationapproachfor patch and tube has the global spatio-temporal location of

thattube. 1) Train a smaller model jointly on images and videos
|     | Specifically, | we  | compute | the | embeddings | as  | follows. | Image |     |     |     |     |     |     |
| --- | ------------- | --- | ------- | --- | ---------- | --- | -------- | ----- | --- | --- | --- | --- | --- | --- |
2D patches
| Hereτ | isaconstanthyperparameter(weused10,000). |     |     |     |     |     | For |     |     |     |     |     |           |         |
| ----- | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------- |
|       |                                          |     |     |     |     |     |     |     |     |     |     |     | S m a l l | C la ss |
jfrom0tod//6(disthenumberoffeatures),andfort,x,y Video ViT  E n c o der ou tp ut
Tube patches
| from0toT,H,W,z |     |     | ∈RT×H×W×D: |     |     |     |     |     |     |     | …   |     |     |     |
| -------------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
i
3) Finetune final layers on
|     |     |           |     |     |     |     |     | 2) Transfer the tubes to a new  |     |     |     |     | the video data. |     |
| --- | --- | --------- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- | --------------- | --- |
|     |     | ω =1/(τj) |     |     |     |     | (3) | large, image pre-trained ViT    |     |     |     |     |                 |     |
j
2D patches from  large model
|     |     | p =sin(t∗ω |     | ),cos(t∗ω |     | )   | (4) |       |     |     |     |     |                   |        |
| --- | --- | ---------- | --- | --------- | --- | --- | --- | ----- | --- | --- | --- | --- | ----------------- | ------ |
|     |     | j,t        |     | j         | j   |     |     | Video |     |     |     |     | Large pre-trained | Class  |
|     |     |            |     |           |     |     |     |       |     |     |     |     | ViT Encoder       | output |
|     |     | p =sin(x∗ω |     | ),cos(x∗ω |     | )   | (5) |       |     |     |     |     |                   |        |
j,x j j T fr u o b m e     s p m at a c l h l  e m s  o a d d e a l pted
…
|                             |             | p =sin(y∗ω      |              | ),cos(y∗ω               |     | )         | (6)    |                                                 |                         |     |                   |                         |            |              |
| --------------------------- | ----------- | --------------- | ------------ | ----------------------- | --- | --------- | ------ | ----------------------------------------------- | ----------------------- | --- | ----------------- | ----------------------- | ---------- | ------------ |
|                             |             | j,y             |              | j                       |     | j         |        |                                                 |                         |     |                   |                         |            |              |
|                             | z [t,x,y,6j |                 | :6(j+1)]+=[p |                         | ,p  | ,p        | ] (7)  |                                                 |                         |     |                   |                         |            |              |
|                             | i           |                 |              |                         | j,t | j,x j,y   |        |                                                 |                         |     |                   |                         |            |              |
|                             |             |                 |              |                         |     |           |        | Figure3.                                        | ScalingofTubeViTmodels: |     |                   | buildinglargescalevideo |            |              |
|                             |             |                 |              |                         |     |           |        | m1)odTreainlinsg oni ismagee axndp videeonsive. |                         | We  | propose           | to expand               | model      | capacity for |
| This                        | adds each   | spatio-temporal |              | position                |     | embedding | to the |                                                 |                         |     |                   |                         |            |              |
|                             |             |                 |              |                         |     |           |        | video models                                    | leveraging              |     | large pre-trained |                         | ViTs. With | TubeViT      |
| featuredimensionofthetokenz |             |                 |              | . Followingpreviouswork |     |           |        |                                                 |                         |     |                   |                         |            |              |
i we can easily train on both image and video dTubae Pattcahes a small-scale
[51],thisisdonefordifferentwavelengthsforeachchannel. modeImla.gTehenwecanadaptthesparsevideotubestoam S um a cl l h larC gla e r
|     |     |     |     |     |     |     |     |     |     |     |     |     |     | ViT  E n c o d er ou tp ss u t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------ |
d//6 is used since we have 6 elements (a sine and cosine imagVeid-eoonlytrainedViT,whichcanbemostlyfrozen.
2D Patches
| valueforeachx,y,t),thiscreatesapositionvalueforeach |     |     |     |     |     |     |     |     |     |     | …   |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
channeloftherepresentation. 2) Sparse Video Tubes  3) Pre-trained large model
|     |              |      |           |            |     |            |        |     |     | adaptation (Space2Depth) |     |     |     | on images only |
| --- | ------------ | ---- | --------- | ---------- | --- | ---------- | ------ | --- | --- | ------------------------ | --- | --- | --- | -------------- |
|     | Importantly, | here | z [t,x,y] | represents |     | the center | of the |     |     |                          |     |     |     |                |
i d4e) Lnargse-escalev msodelm can otrainr espatiallydenseandthedepthtospacefactor
on image, video or both
tube, taking into account any strides and offsets used in (2,4,8,etc.). Tube Patches
the tube construction (the channel dimension is not shown L a rg e  p r e - t r ai n ed
|     |     |     |     |     |     |     |     | InItmeagrep orolatedKernels. |     |     | Forthissetting,rathertha |     |     | n V iT   hE n c a o d ve r - |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------- | --- | --- | ------------------------ | --- | --- | ---------------------------- |
Video
here). ingauniquekernelforeac…htube, welearn2D P1atches3Dkernelof
Afterthetokenizationstep,weconcatenateallthetokens
|     |     |     |     |     |     |     |     | shape8×8×8. |     | Wethenusetri-linearinterpolationtore- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------------------------------------- | --- | --- | --- | --- |
togetherandapplyastandardtransformermodel. Thissim- 2 )   S p a r s e  V i d e o  T u b e s   3 )   P re - tr a in e d   larg e m od el
|     |     |     |     |     |     |     |     | shapethekernel |     | at d a op t a t ivo n  a( S | p ra ci eo 2 D ue p tsh) sizes,e.g.,4x16x |     | o n 1  im 6a g e | s  oo n lr y 3 2 x 4x4, |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | --------------------------- | ----------------------------------------- | --- | ---------------- | ----------------------- |
plestructureletsthemodelsharethemajorityoftheweights
|     |     |     |     |     |     |     |     | etc. depending |     | on the | tube configuration. |     | Any | sized ker- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------ | ------------------- | --- | --- | ---------- |
betweenallinputs,whichwefindtobequitebeneficial.
|     |     |     |     |     |     |     |     | nelcanbecreatedfromthissinglekernel. |     |     |            |     | Thismethodhas |            |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | ---------- | --- | ------------- | ---------- |
|     |     |     |     |     |     |     |     | several advantages.                  |     | (1) | It reduces | the | number        | of learned |
3.4.SparseTubeConstruction
|     |     |     |     |     |     |     |     | parameters | that | are only | used | on the video | stream. | (2) It |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---- | -------- | ---- | ------------ | ------- | ------ |
We explore several methods to create the visual tubes. enables more flexible usage of the kernels, e.g., it can be
Ourcoreapproachconsistof2tubes: the1×16×16×d made longer to handle longer videos, or spatially larger to
| tube | used to | tokenize | the | image and | a 8×8×8×d |     | tube | findsmallobjects. |     |     |     |     |     |     |
| ---- | ------- | -------- | --- | --------- | --------- | --- | ---- | ----------------- | --- | --- | --- | --- | --- | --- |
16×
additionally used for the video. Both have strides of The TubeViT approach consists of the union of the
16×16. Thisbasetokenizerprovidesstrongperformance, above-mentioned Multi-Tube and Space-to-Depth, the ex-
butweexploreseveralvariationsonit.
|     |     |     |     |     |     |     |     | actsettingsareprovidedinthesupplementalmaterials. |     |     |     |     |     | We  |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Multi-Tube.Weaddmultipletubestothecoreapproach experimentwithInterpolatedKernelsinablations.
| ofvarioussizes. |     | Forexample,wecanaddtemporallylong |     |     |     |     |     |     |     |     |     |     |     |     |
| --------------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
andspatiallysmalltubes,suchas16×4×4tolearnlong 3.5.ImageandVideoJointTraining
actions,ormorespatiallyfocusedtubessuchasa2×16×16
| tube. | There | are many | variations | of  | tube | shape | and stride, |              |     |        |              |            |     |           |
| ----- | ----- | -------- | ---------- | --- | ---- | ----- | ----------- | ------------ | --- | ------ | ------------ | ---------- | --- | --------- |
|       |       |          |            |     |      |       |             | As described |     | above, | our approach | seamlessly |     | adapts to |
whichweexperimentallyexplore. eitherimage,videoorbothinputs.Whileimage+videojoint
Space-to-Depth Another way to extend the core ap- inputsarerare,theabilitytousethemtogetherwhiletrain-
proach is a method inspired by depth-to-space [41]. Here, ingisveryimportantasmanydatasetswithvaluableanno-
wereducethenumberofchannelsinatube,e.g.,byafac- tations (e.g., ImageNet, Kinetics) come from either image
|         | ThusthetubeshapebecomesT |     |     |     | ×H  | ×W  | ×d/2. |                                  |     |     |     |                     |     |     |
| ------- | ------------------------ | --- | --- | --- | --- | --- | ----- | -------------------------------- | --- | --- | --- | ------------------- | --- | --- |
| torof2. |                          |     |     |     |     |     |       | sourcesorvideosourcesbutnotboth. |     |     |     | Jointlytrainingwith |     |     |
Next, weconcatenate2tokensalongthechannelaxis. We our approach is easy – the image is tokenized by the 2D
can then also reduce the stride of the tube. This results in kernel and the video is tokenized by both the 2D patches
thesamenumberoftokensanddimensionsastheoriginal, (withlargetemporalstride)andSparseTubes.Botharethen
but effectively increases the kernel size without changing passedintoastandardViT;thepositionembeddingwillbe
the number of parameters. I.e., when the stride is reduced suppliedineithercase.Thepositionembeddingapproachis
onthetimeaxis,thetokennowrepresentsT ∗2×H ×W alsoneededforthejointtrainingtobeeffective.Wedemon-
locations,butonlyusesT∗H∗W parameters.Intheexper- strate the benefits of our approach for joint training in the
iments, we explore different settings: e.g., more temporal experiments,Section4.

3.6.Image-To-VideoScalingUpofModels Method PTData Top1 Top5 Crops TFLOPs
|     |     |     |     |     |     | TSM-ResNeXt-101[26] |     | ImageNet-1k |     | 76.3 – | –   | –   |
| --- | --- | --- | --- | --- | --- | ------------------- | --- | ----------- | --- | ------ | --- | --- |
We also propose a method for a more efficient way of I3DNL[56] ImageNet-1k 77.7 93.3 10×3 10.77
scalingupthemodels(Figure3).TraininglargeViTmodels VidTR-L[70] ImageNet-1k 79.1 93.9 10×3 10.53
is computationally expensive, especially for videos. Since LGD-3DR101[36] ImageNet-lk 79.4 94.4 – –
|     |     |     |     |     |     | SlowFastR101-NL[16] |     |     | -   | 79.8 93.9 | 10×3 | 7.02 |
| --- | --- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --------- | ---- | ---- |
nearlyallthecomponentsofourmodelaresharedbetween X3D-XXL[14] - 80.4 94.6 10×3 5.82
thebothimagesandvideos,weexploreamethodtoutilize OmniSource[12] ImageNet-1k 80.5 94.4 – –
largemodelswithouthavingheavyfine-tuning. TimeSformer-L[6] ImageNet-21k 80.7 94.7 1×3 7.14
|     |     |     |     |     |     | MFormer-HR[33] |     | ImageNet-21k |     | 81.1 95.2 | 10×3 | 28.76 |
| --- | --- | --- | --- | --- | --- | -------------- | --- | ------------ | --- | --------- | ---- | ----- |
First, we train a smaller model jointly on images and MViT-B[13] - 81.2 95.1 3×3 4.10
videos.Thisgivesusasetofweightsforthetubes.Thenwe MoViNet-A6[21] - 81.5 95.3 1×1 0.39
|     |     |     |     |     |     | ViViT-LFE[3] |     | ImageNet-1k |     | 81.7 93.8 | 1×3 | 11.94 |
| --- | --- | --- | --- | --- | --- | ------------ | --- | ----------- | --- | --------- | --- | ----- |
takealargepre-trainedimageViT,butfurtheraddthetubes. MTV-B[62] ImageNet-21K 82.4 95.2 4×3 11.16
These tubes use the same kernel weights as the smaller VideoMAE[47] - 87.4 97.6 - -
model, and so we can avoid further training them. Since LargeScalePretrainingData
larger ViTs generally use more channel dimensions than VATT-L[1] HowTo100M 82.1 95.5 4×3 29.80
|               |        |                    |     |           |       | ip-CSN-152[49] |     | IG-65M |     | 82.5 95.3 | 10×3 | 3.27 |
| ------------- | ------ | ------------------ | --- | --------- | ----- | -------------- | --- | ------ | --- | --------- | ---- | ---- |
| smaller ones, | we use | the space-to-depth |     | transform | again |                |     |        |     |           |      |      |
|               |        |                    |     |           |       | R3D-RS[11]     |     |        | WTS | 83.5 –    | 10×3 | 9.21 |
here to create tokens with the proper channel dimensions OmniSource[12] IG-65M 83.6 96.0 – –
| withoutneedingnewweights. |     |     |     |     |     | MAE-ST[15] |     | IG-1M |     | 84.4 -    | -   | -     |
| ------------------------- | --- | --- | --- | --- | --- | ---------- | --- | ----- | --- | --------- | --- | ----- |
|                           |     |     |     |     |     | ViViT-H[3] |     |       | JFT | 84.9 95.8 | 4×3 | 47.77 |
Next, we pick a point in the network and freeze all the TokenLearner-L/10[38] JFT 85.4 96.3 4×3 48.91
layersbeforeit,forexample,the26thof32layersinViT-H. Florence[65] FLD-900M 86.5 97.3 4×3 –
Atthispoint,weaddagatedconnectiontothenetwork: CoVeR [68] JFT-3B 87.2 – 1×3 –
|     |                           |     |     |     |     | CoCa[64]  |     | ALIGN(1.8B) |     | 88.9 -    | -   | -     |
| --- | ------------------------- | --- | --- | --- | --- | --------- | --- | ----------- | --- | --------- | --- | ----- |
|     |                           |     |     |     |     | MTV-H[62] |     | WTS280p     |     | 89.9 98.3 | 4×3 | 73.57 |
| zs  | =MLP(LN(ys))+ys+tanh(α)z0 |     |     |     | (8) |           |     |             |     |           |     |       |
|     |                           |     |     |     |     | TubeVit-B |     | ImageNet-1k |     | 88.6 97.6 | 4×3 | 0.87  |
|     |                           |     |     |     |     | TubeVit-L |     | ImageNet-1k |     | 90.2 98.6 | 4×3 | 9.53  |
wheresisthelayerthenetworkisfrozenat(e.g.,26)ofthe TubeViT-H(created) ImageNet-1k 90.9 98.9 4×3 17.64
| ViTmodelandz0istherawinputtokensfromthetubes. |     |     |                 |     | α          |          |             |             |      |         |          |       |
| --------------------------------------------- | --- | --- | --------------- | --- | ---------- | -------- | ----------- | ----------- | ---- | ------- | -------- | ----- |
| isthelearnedgatingparameter,                  |     |     | initializedat0. |     | Inthefirst |          |             |             |      |         |          |       |
|                                               |     |     |                 |     |            | Table 1. | Performance | on Kinetics | 400. | TubeViT | performs | best. |
steps of training, this gate has no effect on the representa- We report the crops and total TFLOPs used for inference. The
tion,andthustheViTisunchanged. However,itcanlearn crops,t×xdenotesttemporalandxspatialcrops.
toincorporatetherawtubesatthispointandfurtherrefine
thelaterweights.
|     |     |     |     |     |     | experiments | over | many | tube configurations, |     | as  | well as the |
| --- | --- | --- | --- | --- | --- | ----------- | ---- | ---- | -------------------- | --- | --- | ----------- |
space-to-depthsettingsused.
4.Experiments
Wewouldliketonotethatwithdataaugmentationsuch
We evaluate the approach on several popular datasets: as random spatial and temporal cropping, over multiple
Kinetics400,Kinetics600,Kinetics700[7,20],andSome-
|     |     |     |     |     |     | training | epochs | the model | will | see different |     | parts of the |
| --- | --- | --- | --- | --- | --- | -------- | ------ | --------- | ---- | ------------- | --- | ------------ |
thingSomething V2 [18]. These datasets cover a wide va- video,evenwithsparsesampling.
| riety of video | understanding |     | challenges | and are | well es- |            |     |          |        |            |     |               |
| -------------- | ------------- | --- | ---------- | ------- | -------- | ---------- | --- | -------- | ------ | ---------- | --- | ------------- |
|                |               |     |            |         |          | Comparison |     | to SOTA. | First, | we compare |     | our final ap- |
tablished in the literature. The main results are trained proach to previous state-of-the-art (SOTA) methods. Ta-
| jointly on | ImageNet-1k | (of | 1.2M images) | and | the video |           |       |           |             |     |     |            |
| ---------- | ----------- | --- | ------------ | --- | --------- | --------- | ----- | --------- | ----------- | --- | --- | ---------- |
|            |             |     |              |     |           | bles 1, 2 | and 3 | shows the | performance | of  | our | model com- |
data, pleaseseethesupplementalmaterialsforfulldetails.
|     |     |     |     |     |     | pared to | the state-of-the-art |     | on the | Kinetics-400 |     | Kinetics- |
| --- | --- | --- | --- | --- | --- | -------- | -------------------- | --- | ------ | ------------ | --- | --------- |
We use standard Top 1 and Top 5 evaluation metrics and 600 and Kinetics-700 datasets. Table 1 shows additional
| reportFLOPsofoursandpreviousworks, |           |          |      | whenavailable. |        |             |           |        |              |           |         |           |
| ---------------------------------- | --------- | -------- | ---- | -------------- | ------ | ----------- | --------- | ------ | ------------ | --------- | ------- | --------- |
|                                    |           |          |      |                |        | information | (e.g.     | views, | pre-training | datasets) |         | which ap- |
| Our model                          | sizes are | 90M Base | (B), | 311M Large     | (L). A |             |           |        |              |           |         |           |
|                                    |           |          |      |                |        | plies to    | the other | tables | as well.     | These     | results | show our  |
635MHuge(H)is‘created’withImage-to-Videoscaling. approachoutperformsSOTA,bothintermsofaccuracyand
|     |     |     |     |     |     | efficiency. | We also | outperform |     | methods | on co-training | of  |
| --- | --- | --- | --- | --- | --- | ----------- | ------- | ---------- | --- | ------- | -------------- | --- |
4.1.Mainresults
|         |               |     |             |          |           | images    | and videos, | and | methods | with | strong | video pre- |
| ------- | ------------- | --- | ----------- | -------- | --------- | --------- | ----------- | --- | ------- | ---- | ------ | ---------- |
| For the | main results, | we  | use 4 tubes | with the | following | training. |             |     |         |      |        |            |
configuration(orderoft,h,w): (1)8×8×8withastride We note that all the sizes of our model perform well,
of(16,32,32);(2)16×4×4withastrideof6×32×32 despite the fact that others are much larger or use signifi-
and an offset of (4,8,8); (3) 4×12×12 with a stride of cantly larger pre-training data (e.g., CoCa with 1B params
(16,32,32)andanoffsetof(0,16,16);and(4)1×16×16 and 1.8B examples, MerlotReserve has 644M params and
withastrideof(32,16,16).Foraninputof32×224×224, uses YT-1B dataset). Table 4 shows our results on the
thisresultsinonly559tokens,significantlylessthanother Something-Something dataset (SSv2). This dataset is of-
approaches. Inthesupplementalmaterial,wehavedetailed tenusedtoevaluatemoredynamicactivities. Ourapproach

| Method              | Top1 Top5 |                               |     |     |     |     |     | Kinetics600 |
| ------------------- | --------- | ----------------------------- | --- | --- | --- | --- | --- | ----------- |
| SlowFastR101-NL[16] | 81.8 95.1 | TubeViT-LKinetics-only        |     |     |     |     |     | 85.6        |
| X3D-XL[14]          | 81.9 95.5 | TubeViT-LImageNetthenKinetics |     |     |     |     |     | 90.4        |
TimeSformer-L[6] 82.2 95.6 TubeViT-LImageNet+KineticsJointly 91.5
| MFormer-HR[33] | 82.7 96.1 |                                       |     |     |     |     |     |      |
| -------------- | --------- | ------------------------------------- | --- | --- | --- | --- | --- | ---- |
|                |           | 2DPatchesonlyImageNet+Kinetics        |     |     |     |     |     | 87.6 |
| ViViT-LFE[3]   | 82.9 94.6 |                                       |     |     |     |     |     |      |
|                |           | Inflated3DPatchesImageNetthenKinetics |     |     |     |     |     | 88.4 |
| MViT-B[13]     | 83.8 96.3 |                                       |     |     |     |     |     |      |
| MoViNet-A6[21] | 84.8 96.5 |                                       |     |     |     |     |     |      |
Table5. Combiningdatasets,whichTubeViTseamlesslyallows,
R3D-RS[11](WTS) 84.3 – ishighlyeffective,asseenhereintheseside-by-sideresultsforthe
ViViT-H[3](JFT) 85.8 96.5 Kinetics-600dataset.TheresultsarebasedontheViT-Lmodel.
| TokenLearner-L/10[38](JFT) | 86.3 97.0 |     |     |     |     |     |     |     |
| -------------------------- | --------- | --- | --- | --- | --- | --- | --- | --- |
| Florence[65](FLD-900M)     | 87.8 97.8 |     |     |     |     |     |     |     |
CoVeR[68](JFT-3B) 87.9 – behighlyeffectiveasalsoshownabove. Table5evaluates
|                    |           | this in a | side-by-side | experiment |     | of using | Kinetics | (video) |
| ------------------ | --------- | --------- | ------------ | ---------- | --- | -------- | -------- | ------- |
| MTV-H[62](WTS280p) | 90.3 98.5 |           |              |            |     |          |          |         |
CoCa[64](ALIGN1.8B) 89.4 - onlyvsKineticsandImageNetdatasetsforpre-training.We
Merlot-Reserve-L[66](YT-1B) 91.1 97.1 seethatthereisalargegainfromtheco-trainingofourap-
proach. Weseethattwo-stagetraining,i.e.,firsttrainingon
| TubeVit-B(ImageNet-1k) | 90.9 97.3 |     |     |     |     |     |     |     |
| ---------------------- | --------- | --- | --- | --- | --- | --- | --- | --- |
onedatasetandthentrainingonasecondone,isalsoweaker
| TubeVit-L(ImageNet-1k) | 91.5 98.7 |          |                 |     |         |          |        |          |
| ---------------------- | --------- | -------- | --------------- | --- | ------- | -------- | ------ | -------- |
|                        |           | than the | joint training, | as  | the two | datasets | cannot | interact |
‘TubeVit-H(created) 91.8 98.9 duringtraining. Wealsocomparetopriormethodssuchas
TimeSFormer[6]onlyusingdense2Dpatches,orusingin-
Table2. PerformanceonKinetics600. Similarly,toTable1our flated3Dkernels(e.g.,ViViT[3]). Inbothcases,weseea
model uses the ImageNet-1k dataset. Most models use signifi- clearbenefitfromtheproposedapproach. Wealsonotethat
cantlylargerpre-trainingdatasets(bottomhalf).Tube-ViToutper- thesepriorapproacheshavesignificantlymoreFLOPs,due
formspriorwork.
|     |     | tothelargenumberoftokensfromthedensesampling. |      |       |           |             |     | Our           |
| --- | --- | --------------------------------------------- | ---- | ----- | --------- | ----------- | --- | ------------- |
|     |     | observations                                  | that | image | and video | co-training |     | is beneficial |
Top1 Top5 areconsistentwithpriorworks[25,68];herethedifference
| VidTR-L[70] | 70.2 – |     |     |     |     |     |     |     |
| ----------- | ------ | --- | --- | --- | --- | --- | --- | --- |
isthatwehaveasinglecompactmodeltodothat.
| SlowFastR101[16] | 71.0 89.6 |     |     |     |     |     |     |     |
| ---------------- | --------- | --- | --- | --- | --- | --- | --- | --- |
MoViNet-A6[21] 72.3 – Asasanitycheck,wealsocompareourperformanceon
CoVeR(JFT-3B)[68] 79.8 – ImageNet-1k, without any hyperparameter tuning or addi-
CoCa(Align1.8B)[64] 82.7 - tions: ourViT-BmodelonlytrainedonImageNethas78.1
| MTV-H(WTS280p)[62] | 83.4 96.2 |           |         |        |       |          |      |              |
| ------------------ | --------- | --------- | ------- | ------ | ----- | -------- | ---- | ------------ |
|                    |           | accuracy, | similar | to the | ViT-B | in [44]. | When | joint train- |
| TubeViT-L          | 83.8 96.6 |           |         |        |       |          |      |              |
ingwithKinetics-600,themodelgets81.4,againof3.4%,
|     |     | showing | the benefits | of  | joint training | for | image-only | tasks |
| --- | --- | ------- | ------------ | --- | -------------- | --- | ---------- | ----- |
Table3.PerformancecomparedtoSOTAonKinetics700.
too.WhileotherworksachievehigherperformanceonIma-
geNet,theyoftenusespecializeddataaugmentation,learn-
Top1 Top5
|                      |     | ingschedules,andothertrickswhichwearenotusing. |     |     |     |     |     | In- |
| -------------------- | --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- |
| SlowFastR50[16] 61.7 | –   |                                                |     |     |     |     |     |     |
TimeSformer-L[6] 62.5 stead, we are purely studying the benefit from using both
| VidTR-L[70] 63.0 | –   |     |     |     |     |     |     |     |
| ---------------- | --- | --- | --- | --- | --- | --- | --- | --- |
videosandimages.
| CoVeR[68] 64.7 | –   |     |     |     |     |     |     |     |
| -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
MoViNet-A3[21] 64.1 88.8 Scalingvideotrainingwithsparsevideotubes. InTa-
| ViViT-LFE[3] 65.9 | 89.9 |          |             |     |         |         |       |        |
| ----------------- | ---- | -------- | ----------- | --- | ------- | ------- | ----- | ------ |
|                   |      | ble 6 we | demonstrate | how | a small | TubeViT | model | can be |
| VoV3D-L[22] 67.3  | 90.5 |          |             |     |         |         |       |        |
MFormer-L[33] 68.1 91.2 adapted leveraging a large and (often independently) pre-
| MTV-B(320p)[62] 68.5 | 90.4 |         |          |        |       |          |               |     |
| -------------------- | ---- | ------- | -------- | ------ | ----- | -------- | ------------- | --- |
|                      |      | trained | model on | images | only. | We start | by leveraging | a   |
| MViT-B[13] 68.7      | 91.5 |         |          |        |       |          |               |     |
MViT[23] 73.3 94.1 large, image-pretrained ViT, here ViT-H. We then take the
| MaskFeat[59] 75.0 | 95.0 |     |     |     |     |     |     |     |
| ----------------- | ---- | --- | --- | --- | --- | --- | --- | --- |
VideoMAE[47] 75.4 95.2 learnedtubesfromTubeViT-Bandusethemalongwiththe
|                |      | ViT-H image         | tokenizer |                               | to generate | a set | of tokens | from a |
| -------------- | ---- | ------------------- | --------- | ----------------------------- | ----------- | ----- | --------- | ------ |
| TubeViT-L 76.1 | 95.2 |                     |           |                               |             |       |           |        |
|                |      | video,sameasbefore. |           | ThentheseareusedasinputtoViT- |             |       |           |        |
Table4.PerformanceonSomething-SomethingV2dataset. H,andwefinetuneonlythelatterpartsofthemodelonthe
|     |     | video data. | These | results | suggests | that | this is | an effective |
| --- | --- | ----------- | ----- | ------- | -------- | ---- | ------- | ------------ |
waytoscaleandutilizegiantViTmodelswithoutneeding
outperformsSOTAonitaswell. thehighcomputecosttofullyfinetunethemodel. Wealso
Jointimage+videotraining. Wefurtherexploretheef- seethatthegatinginEq. 8iseffective. Wealsofoundthat
fectsofco-trainingonimage+videodatasets,findingthisto inthissetting,trainingtimewasreducedby43%,asithas

| Models | K600,Accuracy(%) |     | 4.2.Ablations |     |     |     |     |     |     |     |
| ------ | ---------------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
TubeViT-HFullFinetune 91.8 In this section, we present a number of ablation studies
Scalingmethodwithdifferentportionstrained todeterminewhythismethodiseffective. Fortheseexper-
imentsweuseKinetics600.
| LastFCLayer |     | 85.6 |     |     |     |     |     |     |     |     |
| ----------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
+Last4Layers 86.3 Mainablations. First,westudytheeffectofthechoice
+Last8Layers 86.8 ofpositionbiases(Table7a). Wefindthataddingfixedco-
sinepositionembeddingperformsbestandmuchbetterthan
| +Last8+Gated(Eq. | 8)  | 89.7 |                   |          |              |             |             |       |         |          |
| ---------------- | --- | ---- | ----------------- | -------- | ------------ | ----------- | ----------- | ----- | ------- | -------- |
|                  |     |      | other embeddings. |          | Intuitively, |             | this        | makes | sense,  | since we |
|                  |     |      | are sparsely      | sampling |              | potentially | overlapping |       | tokens, | this     |
Table6.Image-to-VideoScaling.WetakeaImageNetpre-trained
ViT-H and use a set of Tubes from TubeViT-B to create the to- methodisabletobestcapturethetokenlocation.
kens.Wethenfine-tunedifferentportionsofthemodeltoseehow Next in Table 7b, we study the number of tubes used.
wecanbesttakeadvantageofexisting,largepretrainedViTmod- This finding, which is consistent with previous multi-view
els.Evenpretrainingofhandfuloflayerscanachieveperformance observations [62], shows that having a variety of tubes is
approachingthefullmodeltraining. beneficialtovideounderstanding.
|     |     |     | Next,                                      | inTable7c, | westudythedepth-to-spaceversions |           |     |          |     |          |
| --- | --- | --- | ------------------------------------------ | ---------- | -------------------------------- | --------- | --- | -------- | --- | -------- |
|     |     |     | of the network.                            |            | Here,                            | we reduce | the | channels | of  | the gen- |
|     |     |     | eratedtokensfromD//S,e.g.,byafactorof2or4. |            |                                  |           |     |          |     | Then     |
aftergeneratingthetokens,weconcatenatethemalongthe
|     |     |     | channel    | axis. We | study   | both         | increasing | the         | number | of to-  |
| --- | --- | --- | ---------- | -------- | ------- | ------------ | ---------- | ----------- | ------ | ------- |
| 86  |     |     | kens along | the      | spatial | and temporal |            | dimensions. |        | We find |
ycaruccA 006-sciteniK thistobeaneffectivemethod,asitenablesmoredensesam-
pleswithoutincreasingthenumberofparametersortokens.
84
Table7dcomparesevaluatingwithmorepatchesthanthe
|     |     |     | modelwastrainedwith. |     |     | Todothiswereducethestridesof |     |     |     |     |
| --- | --- | --- | -------------------- | --- | --- | ---------------------------- | --- | --- | --- | --- |
82
|     |     |     | thekernel.  | Initiallythisimprovesresults,butafterincreas- |     |        |          |        |         |     |
| --- | --- | --- | ----------- | --------------------------------------------- | --- | ------ | -------- | ------ | ------- | --- |
|     |     |     | ing 2x, the | performance                                   |     | begins | to drop, | likely | because | the |
80
|     |     | Kinetics+ImageNet | evaluationdataistoodifferentfromthetrainingone. |     |     |     |     |     |     |     |
| --- | --- | ----------------- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Kinetics-Only
InTable7e,westudytheabilityoftheinterpolatedsingle
0 500 1000 1500 2000 2500 kernel. I.e.,ratherthanhavingN 3Dconvolutionalkernels,
Number of Tokens
oneforeachtube,webuild18×8×83Dkernelandusein-
Figure 4. Accuracy vs. Number of tokens used in our model. terpolationtogeneratethedifferenttubeshapes. Somewhat
We find that when increasing the tokens above 1500, there is a surprisingly, we find this works fairly well, while also re-
noticeabledropinperformance,especiallywhenonlytrainingon
ducingthenumberoflearnableparametersinthenetwork.
Kinetics-600data.Jointtrainingismorerobust.
|     |     |     | In Table | 7f,         | we compare |              | the approach |     | with | different |
| --- | --- | --- | -------- | ----------- | ---------- | ------------ | ------------ | --- | ---- | --------- |
|     |     |     | number   | of temporal | and        | spatial      | crops.       | We  | find | that even |
|     |     |     | a single | crop gives  | strong     | performance, |              | and | the  | standard  |
4×3performsnearlythesameasthe10×10setting,sug-
fewerweightstoupdate.
gestingthatthesparsesamplesarequitesuitableandfurther
Detrimental Effects of Too Many Tokens. Next we informationisnotasbeneficial.
study the effect of number of tokens used in the model, Factorized attention ablations. In Table 8, we further
shown in Figure 4. This result is another key insight as study the effect of adding a new attention layer to an Ima-
towhyourapproachworkssowell: withtoomanytokens, geNet pre-trained ViT model. Here, we are using the tube
theperformancedrops,especiallywhenonlyusingKinetics method to tokenize the inputs, but instead of using a fac-
data. There are a number of possible reasons for why this torizedattentionmodule,wesimplyaddanadditionalself-
occurs,forexample,theself-attentionmechanismcouldbe attention layer. This has a similar effect of the factorized
struggling to learn for longer sequences, or there may not attention approaches that add new, uninitialized K,Q,V
besufficientdatatolearnthelongersequences,orperhaps projections to a pre-trained ViT (e.g., TimeSFormer and
themodelisoverfittingwithlongersequences. Thisresult ViViT). These results indicate that such methods are not
indicates that for current datasets, the sparse sampling is abletobestutilizetheimagepre-trainedweightsofthenet-
an effective and efficient way to process videos. Further, workduetothesenewlayers. Sincethesparsetubesyield
it is possible that existing using long, densely sampled se- few additional tokens, they can directly use the same ViT
quencesareeffectedbythis,andperhapsanotherreasonthe modelwithoutfactorizedattentionandarethusabletobet-
factorizedattentionmodulesareneeded. terutilizetheimagetrainedweights.Notethattherearestill

GF K600
|     |     |                        |     |     | K600      |     | GF                | K600 |     |                  |     |                |     |     |
| --- | --- | ---------------------- | --- | --- | --------- | --- | ----------------- | ---- | --- | ---------------- | --- | -------------- | --- | --- |
|     |     |                        |     |     |           |     |                   |      |     | Baseline         |     | 72 83.4        |     |     |
|     |     | None                   |     |     | 78.6      |     | 1 70              | 78.4 |     |                  |     |                |     |     |
|     |     |                        |     |     |           |     |                   |      |     | WithD2Sx2T       |     | 72 84.7        |     |     |
|     |     | Learned                |     |     | 79.2      |     | 2 71              | 81.5 |     | WithD2Sx2S       |     | 72 84.5        |     |     |
|     |     | Relative               |     |     | 77.5      |     |                   |      |     |                  |     |                |     |     |
|     |     |                        |     |     |           |     | 4 72              | 83.4 |     | WithD2Sx4T       |     | 72 85.1        |     |     |
|     |     | FixedCosine(nostride)  |     |     | 77.7      |     |                   |      |     |                  |     |                |     |     |
|     |     |                        |     |     |           |     |                   |      |     | WithD2Sx4S       |     | 72 85.4        |     |     |
|     |     | FixedCosine(Ours)      |     |     | 84.5      |     | 8 74              | 85.4 |     |                  |     |                |     |     |
|     |     |                        |     |     |           |     |                   |      |     | WithD2Sx4ST      |     | 72 85.3        |     |     |
|     |     | (a)PositionEmbeddings. |     |     | Fixed,co- |     | (b)NumberofTubes. |      |     |                  |     |                |     |     |
|     |     |                        |     |     |           |     |                   |      |     | (c)SpaceToDepth. |     | Applyingspace- |     |     |
sineembeddingswithstridesisbest.
to-depthtemporally(T),spatially(S),
andspatio-temporally(ST).
K600
K600
|     |     |                            |              |      |            |     |                             |              | K600    |     |                | 1×1   | 82.8        |     |
| --- | --- | -------------------------- | ------------ | ---- | ---------- | --- | --------------------------- | ------------ | ------- | --- | -------------- | ----- | ----------- | --- |
|     |     |                            | Base(559)    | 84.5 |            |     |                             |              |         |     |                |       |             |     |
|     |     |                            |              |      |            |     |                             |              |         |     |                | 4×1   | 83.3        |     |
|     |     |                            | 768          | 84.9 |            |     |                             | Interpolated | 83.8    |     |                |       |             |     |
|     |     |                            |              |      |            |     |                             |              |         |     |                | 1×3   | 83.6        |     |
|     |     |                            | 1024         | 84.6 |            |     |                             | TubeViT      | 84.5    |     |                |       |             |     |
|     |     |                            |              |      |            |     |                             |              |         |     |                | 4×3   | 84.5        |     |
|     |     |                            | 1536         | 83.5 |            |     |                             |              |         |     |                |       |             |     |
|     |     |                            |              |      |            |     | (e)                         | Interpolated | Kernel. |     |                | 10×10 | 84.7        |     |
|     |     | (d)                        | Eval Tokens. |      | Generating |     | Usingasingle3Dkernelin-     |              |         |     |                |       |             |     |
|     |     |                            |              |      |            |     |                             |              |         |     | (f) Multi-Crop |       | Evaluation. |     |
|     |     | largernumberoftokensateval |              |      |            |     | terpolatedtodifferentsizes. |              |         |     |                |       |             |     |
4×3isusedinthepaper.
|     |     | time | than | in training, | where |     |     |     |     |     |     |     |     |     |
| --- | --- | ---- | ---- | ------------ | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
559areused.
Table7.AblationstudiesonvariouscomponentsofourapproachonKinetics-600,usingTubeViT-B.
|          |         | LayersAdded |      | K600     |       |                 |     |          |                   |             |              |             |               |           |
| -------- | ------- | ----------- | ---- | -------- | ----- | --------------- | --- | -------- | ----------------- | ----------- | ------------ | ----------- | ------------- | --------- |
|          |         |             |      |          |       |                 |     |          |                   | Trained     |              | K600        |               |           |
|          |         | 0           |      | 84.23    |       |                 |     |          |                   | LastFCLayer |              | 79.6        |               |           |
|          |         |             |      |          |       |                 |     |          |                   | +1Layer     |              | 80.8        |               |           |
|          |         | 1           |      | 80.23    |       |                 |     |          |                   |             |              |             |               |           |
|          |         | 2           |      | 78.87    |       |                 |     |          |                   | +4Layers    |              | 81.1        |               |           |
|          |         |             |      |          |       |                 |     |          |                   | WholeModel  |              | 81.4        |               |           |
|          |         | 4           |      | 75.24    |       |                 |     |          |                   |             |              |             |               |           |
|          |         | 8           |      | 72.95    |       |                 |     |          |                   |             |              |             |               |           |
|          |         |             |      |          |       |                 |     | Table    | 9. Image-to-Video |             | scaling from | Tiny        | to Base.      | We take a |
|          |         |             |      |          |       |                 |     | ImageNet | pre-trained       | ViT-Base    | and          | the TubeViT | corresponding |           |
| Table 8. | We find | that adding | even | a single | layer | to a pretrained |     |          |                   |             |              |             |               |           |
imagenetworkdegradesperformance. Thissuggeststhatthefac- toViT-TinyandImageNetpre-trainedViT-Basetocreatealarger
torizedattentionmethodsaresub-optimalsincetheycannotfully TubeViT.Thesemodelsweretrainedfor50ksteps.
takeadvantageoftheimage-pre-trainednetworks.Trainedfor70k
| steps. |     |     |     |     |     |     |     | 2D Patches |     |     |     |     |     |     |
| ------ | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
differencesbetweentheworks,e.g.,thereducednumberof
| tokens, etc. | However, |             | we believe | this    | observation         |     | holds, | 8x8x8 Tube |     |     |     |     |     |     |
| ------------ | -------- | ----------- | ---------- | ------- | ------------------- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- |
| and is a     | possible | explanation |            | for why | the spatio-temporal |     |        |            |     |     |     |     |     |     |
attentioninViVitperformedbetterforsomedatasets.
| Modelscalingablations.                  |           |             |     | Table9providesablationson |             |          |         |              |     |     |     |     |     |     |
| --------------------------------------- | --------- | ----------- | --- | ------------------------- | ----------- | -------- | ------- | ------------ | --- | --- | --- | --- | --- | --- |
| scalingtocreateTubeViTBasefromaTinyone. |           |             |     |                           |             | Evenjust |         | 4x12x12 Tube |     |     |     |     |     |     |
| training                                | the final | few layers  | is  | effective                 | (4 of       | 12), and | can     |              |     |     |     |     |     |     |
| nearly match                            | the       | performance |     | of full                   | finetuning. |          | This is |              |     |     |     |     |     |     |
consistentwithourobservationsinTable6forViT-H.
Figure5.Visualizationofaselectedsetof2Dpatchesandtubes.
Figure5visualizesthelearned2Dpatchesand3Dtubes.
5.Conclusion also demonstrate an elegant scaling of video models with
|             |     |        |       |       |           |              |     | our proposed |     | method.   | We conduct | extensive | ablation | ex-     |
| ----------- | --- | ------ | ----- | ----- | --------- | ------------ | --- | ------------ | --- | --------- | ---------- | --------- | -------- | ------- |
| We proposed |     | sparse | video | tubes | for video | recognition. |     |              |     |           |            |           |          |         |
|             |     |        |       |       |           |              |     | periments    | to  | determine | why the    | approach  | works,   | finding |
Withsparsevideotubes,aViTencodercanbetransformed
theacombinationofthejointtraining,reducedtokens,and
| into an efficient |     | video model. |     | The approach |     | is simple, | en- |     |     |     |     |     |     |     |
| ----------------- | --- | ------------ | --- | ------------ | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
betterutilizationofsharedimage+videoweightsledtothe
| ables seamless |       | joint training |        | with images | and      | videos    | and |               |     |                                 |     |     |     |     |
| -------------- | ----- | -------------- | ------ | ----------- | -------- | --------- | --- | ------------- | --- | ------------------------------- | --- | --- | --- | --- |
|                |       |                |        |             |          |           |     | improvements. |     | WeobtainSOTAoraboveperformance. |     |     |     |     |
| improves       | video | recognition    | across |             | multiple | datasets. | We  |               |     |                                 |     |     |     |     |

| A.ImplementationDetails |                                       |     |                |     |     |       |         | • 32×8×8  |     |     |     |     |     |     |
| ----------------------- | ------------------------------------- | --- | -------------- | --- | --- | ----- | ------- | --------- | --- | --- | --- | --- | --- | --- |
| Our hyperparameters     |                                       |     | are summarized |     | in  | Table | 10. For | • 4×32×32 |     |     |     |     |     |     |
| alldatasets,            | weemployrandomspatialandtemporalcrop- |     |                |     |     |       |         |           |     |     |     |     |     |     |
ping. Formostdatasets, thesesettingswerethesame. For Wenotetwoimportantfactors. First,sinceweuseinter-
Charades,wedecreasedthebatchsizebutusedlonger,128 polationtocreatethelargerkernels,thenumberoflearned
frame clips, as Charades videos are roughly 30 seconds parameters is the same, and initialized from the same ker-
|     |     |     |     |     |     |     |     | nels for | the other datasets. | Second, | since | the | number | of  |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------------------- | ------- | ----- | --- | ------ | --- |
long,comparedto10secondsforKinetics.
We also found some training instability when using stridesisunchanged,thisresultsisthesamenumberofto-
largerViTmodels. WhenusingViT-LorViT-Hmodels,we kens. Critically,thischangehasverylittleeffectonthenet-
hadtodecreasetheweightdecayvalueaswellasthelearn- work and its parameters, but enables the model to better
ingrate,otherwisewefoundthetrainingaccuracydropped capturetheinformationforCharades.
|     |     |     |     |     |     |     |     | In Table | 13, we report | the results. |     | The core | MultiTube |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------------- | ------------ | --- | -------- | --------- | --- |
to0andthelossstayedflat.
Forsmallerdatasets,suchasCharadesandSSv2,wehad approachperformsquitewell,butwiththeinterpolatedker-
toincreasethedataaugmentationsettings,asdoneinprevi- nels,isabletoperformonparwithTokenLearner[38],the
ousworks,e.g.,[62].WeaddedMixupandlabelsmoothing state-of-the-art,whilestillsparselysamplingthevideo. We
anddropouttothem. also perform similarly using significantly less data, e.g.,
Forallthedatasets,weappliedRandAugment[9],aswe JFT-300M was used to pre-trained TokenLearner, we ac-
foundthistobebeneficial.Wealsokeptthenumberofsteps complish the same performance without such large scale
| thesameforalldatasets. |          |     |          |           |     |      |         | data. |     |     |     |     |     |     |
| ---------------------- | -------- | --- | -------- | --------- | --- | ---- | ------- | ----- | --- | --- | --- | --- | --- | --- |
| Joint                  | ImageNet | and | Kinetics | Training. |     | When | jointly |       |     |     |     |     |     |     |
training on the two (or more) datasets, we use a separate C.AblationsonTubeShapes.
| fully connected |     | layer to | output | the class | predictions. |     | E.g., |     |     |     |     |     |     |     |
| --------------- | --- | -------- | ------ | --------- | ------------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
InTable14,weprovideadetailedstudyonKinetics-600
| for ImageNet                               | and      | Kinetics-600, |         | we                   | use an | FC layer | with    |                                         |                      |              |            |              |              |        |
| ------------------------------------------ | -------- | ------------- | ------- | -------------------- | ------ | -------- | ------- | --------------------------------------- | -------------------- | ------------ | ---------- | ------------ | ------------ | ------ |
|                                            |          |               |         |                      |        |          |         | of various                              | tube configurations. |              | We observe | that         | the          | model  |
| 1,000 and                                  | 600      | outputs.      | We then | compute              | the    | loss     | for the |                                         |                      |              |            |              |              |        |
|                                            |          |               |         |                      |        |          |         | isn’t overly                            | sensitive to         | tube shapes, | at         | least        | on Kinetics- |        |
| relevantheadandbackpropagateit.            |          |               |         | Duringthejointtrain- |        |          |         |                                         |                      |              |            |              |              |        |
|                                            |          |               |         |                      |        |          |         | 600, but                                | having multiple,     | different    | tubes,     | as well      | as           | varia- |
| ing,weusethesamesettingsaslistedinTable10. |          |               |         |                      |        |          | Weuse   |                                         |                      |              |            |              |              |        |
|                                            |          |               |         |                      |        |          |         | tionintheirshapesisgenerallybeneficial. |                      |              |            | Weusethefol- |              |        |
| the joint                                  | training | for Kinetics  |         | 400, 600             | and    | 700. For | Cha-    |                                         |                      |              |            |              |              |        |
lowingtubesintheseexperiments:
| rades and | SSv2, | we use | the | Kinetics-600+ImageNet |     |     | pre- |     |     |     |     |     |     |     |
| --------- | ----- | ------ | --- | --------------------- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
trainedmodelandfinetuneitonthedataset.
(a) 1×16×16
FullModelSettings.Ourmodelisbasedonthestandard
| ViT models, | thus       | the core | of        | the approach |          | is the | same as  | (b) 4×32×32 |     |     |     |     |     |     |
| ----------- | ---------- | -------- | --------- | ------------ | -------- | ------ | -------- | ----------- | --- | --- | --- | --- | --- | --- |
| previous    | ViTs [10]. | We       | summarize | those        | settings |        | in Table |             |     |     |     |     |     |     |
(c) 4×4×4
11.
InTable12,wedetailthesettingsforeachtube.
(d) 4×12×12
| B.AdditionalExperimentsonCharades     |             |          |          |      |                  |               |        | (e) 8×8×8    |     |     |     |     |     |     |
| ------------------------------------- | ----------- | -------- | -------- | ---- | ---------------- | ------------- | ------ | ------------ | --- | --- | --- | --- | --- | --- |
| We include                            | results     | on       | Charades | [42] | to show          | the           | effec- | (f) 16×4×4   |     |     |     |     |     |     |
| tivenessofthisapproachonlongervideos, |             |          |          |      |                  | sinceCharades |        |              |     |     |     |     |     |     |
| videosareonaverage30secondslong.      |             |          |          |      | However,Charades |               |        | (g) 16×16×16 |     |     |     |     |     |     |
| is also a                             | multi-label | dataset, | and      | we   | found            | it required   | dif-   |              |     |     |     |     |     |     |
(h) 32×8×8
| ferent settings | to                                      | effectively | train,   | so         | we include |          | all those |                         |     |     |     |     |     |     |
| --------------- | --------------------------------------- | ----------- | -------- | ---------- | ---------- | -------- | --------- | ----------------------- | --- | --- | --- | --- | --- | --- |
| detailshere.    |                                         |             |          |            |            |          |           | andthefollowingstrides: |     |     |     |     |     |     |
| First,          | we found                                | that        | the core | multi-tube |            | approach | were      |                         |     |     |     |     |     |     |
| not performing  |                                         | as well     | as some  | prior      | work       | (e.g.,   | Assem-    | (i) (4,16,16)           |     |     |     |     |     |     |
| bleNet[39]).    | SinceCharadeshasalotofobject-relatedac- |             |          |            |            |          |           |                         |     |     |     |     |     |     |
(ii) (8,8,8)
tionsandcontainslongervideoswithmoretemporalinfor-
| mation, we               | modified | the                                  | core | model | to make | it more | suit- | (iii) (8,32,32) |     |     |     |     |     |     |
| ------------------------ | -------- | ------------------------------------ | ---- | ----- | ------- | ------- | ----- | --------------- | --- | --- | --- | --- | --- | --- |
| ableforthisdata.         |          | First,weusedtheinterpolationmethodto |      |       |         |         |       |                 |     |     |     |     |     |     |
| increasethetubeshapesto: |          |                                      |      |       |         |         |       | (iv) (16,16,16) |     |     |     |     |     |     |
(v) (32,32,32)
• 1×16×16
• 16×16×16

|     |     |     |     | K400 | K600 | K700 |     | Charades | SSv2 |
| --- | --- | --- | --- | ---- | ---- | ---- | --- | -------- | ---- |
Optimization
| Optimizer            |     |     |     |     |                          |     | Adam    |      |      |
| -------------------- | --- | --- | --- | --- | ------------------------ | --- | ------- | ---- | ---- |
| Batchsize            |     |     |     | 256 |                          | 256 | 256     | 64   | 256  |
| Learningrateschedule |     |     |     |     | cosinedecay+linearwarmup |     |         |      |      |
| Linearwarmupsteps    |     |     |     |     |                          |     | 10,000  |      |      |
| Baselearningrate     |     |     |     |     | 5e-5(L,H1e-5)            |     |         | 1e-3 | 2e-5 |
| Steps                |     |     |     |     |                          |     | 300,000 |      |      |
Dataaugmentation
| Randaugmentnumberoflayers[9] |     |     |     |     |     |                   | 2   |     |     |
| ---------------------------- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- |
| Randaugmentmagnitude[9]      |     |     |     |     |     |                   | 10  |     |     |
| WeightDecay                  |     |     |     |     |     | 0.001(B)1e-5(L,H) |     |     |     |
| Mixup[69]                    |     |     |     |     | -   | -                 | -   | -   | 0.3 |
| Dropout                      |     |     |     |     | -   | -                 | -   | 0.2 | 0.3 |
| LabelSmoothing               |     |     |     |     | -   | -                 | -   | 0.1 | 0.3 |
| NumberofFrames               |     |     |     | 64  |     | 64                | 64  | 128 | 32  |
| FPS                          |     |     |     | 15  |     | 15                | 15  | 6   | 24  |
Table10.Traininghyperparamtersforourexperiments.Wenotewhendifferentsettingswereusedforthebase(B),large(L)andhuge(H)
models.
|     | Model     | Layers | Hiddensized |     | MLPsize |     | NumHeads | Params |      |
| --- | --------- | ------ | ----------- | --- | ------- | --- | -------- | ------ | ---- |
|     | ViT-Base  | 12     | 768         |     | 3072    |     | 12       |        | 86M  |
|     | ViT-Large | 24     | 1024        |     | 4096    |     | 16       |        | 307M |
|     | ViT-Huge  | 32     | 1280        |     | 5120    |     | 16       |        | 632M |
Table11.Parametercountforthevitencoderbackbones.
|     | Kernel  |     | Stride     | Offset    |     | S2D        |     | params |     |
| --- | ------- | --- | ---------- | --------- | --- | ---------- | --- | ------ | --- |
|     | 8×8×8   |     | (16,32,32) | (0,0,0)   |     | 2xtemporal |     | 512d   |     |
|     | 16×4×4  |     | (6,32,32)  | (4,8,8)   |     | 4xspatial  |     | 256d   |     |
|     | 4×12×12 |     | (16,32,32) | (0,16,16) |     |            | -   | 576d   |     |
|     | 1×16×16 |     | (32,16,16) | (0,0,0)   |     |            | -   | 256d   |     |
Table12.ConfigurationforthetubesusingtheinmainTube-ViT.Wealsoreportthenumberofparamsusedbyeachtube,whichdepends
ond,thehiddensizeoftheViTmodelused. Thetubesaddanadditional1-3Mparams,dependingonthemodel,asmallfractionofthe
totalmodelsize.

mAP
SlowFast[16] 45.2
AssembleNet-101[40] 58.6
AssembleNet++-50[39] 59.8
MoViNet-A6[21] 63.2
TokenLearner[38] 66.3
MultiTubeTube-ViT-L 61.8
InterpolatedTubeViT-L 66.2
Table13.Charadesclassification.
TubeConfig K600
(a+iv)+(b+v)+(f+iv) 87.9
(c+iv)+(e+v)+(g+iv) 87.5
(a+iv)+(e+v)+(g+iv) 87.8
(b+iv)+(e+v)+(g+iv) 87.7
(a+iv)+(d+v)+(e+iv)+(h+v) 88.6
(a+iv)+(b+v)+(c+iv)+(h+v) 87.9
(a+iv)+(d+v)+(e+iv)+(f+v) 88.9
Table14.Ablationondifferenttubeshapes,trainedfor50ksteps.

References [16] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and
|     |     |     |     |     |     |     | Kaiming | He. | Slowfast | networks | for | video | recognition. | In  |
| --- | --- | --- | --- | --- | --- | --- | ------- | --- | -------- | -------- | --- | ----- | ------------ | --- |
[1] Hassan Akbari, Linagzhe Yuan, Rui Qian, Wei-Hong ICCV,2019. 1,2,3,5,6,11
| Chuang,Shih-FuChang,YinCui,andBoqingGong. |     |            |                 |     |          | Vatt: |               |                                          |               |            |           |            |               |          |
| ----------------------------------------- | --- | ---------- | --------------- | --- | -------- | ----- | ------------- | ---------------------------------------- | ------------- | ---------- | --------- | ---------- | ------------- | -------- |
|                                           |     |            |                 |     |          |       | [17] Shreyank | N                                        | Gowda, Marcus |            | Rohrbach, | and        | Laura         | Sevilla- |
| Transformers                              | for | multimodal | self-supervised |     | learning | from  |               |                                          |               |            |           |            |               |          |
|                                           |     |            |                 |     |          |       | Lara.         | Smartframeselectionforactionrecognition. |               |            |           |            |               | InPro-   |
| rawvideo,audioandtext.                    |     |            | InNeurIPS,2021. |     | 1,5      |       |               |                                          |               |            |           |            |               |          |
|                                           |     |            |                 |     |          |       | ceedings      | of                                       | the AAAI      | Conference | on        | Artificial | Intelligence, |          |
[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, An- volume35,pages1451–1459,2021. 2
toine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur [18] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michal-
| Mensch,           | Katie | Millican, | Malcolm   | Reynolds, | Roman         | Ring, |                  |         |              |         |           |            |       |        |
| ----------------- | ----- | --------- | --------- | --------- | ------------- | ----- | ---------------- | ------- | ------------ | ------- | --------- | ---------- | ----- | ------ |
|                   |       |           |           |           |               |       | ski,             | Joanna  | Materzynska, | Susanne |           | Westphal,  | Heuna | Kim,   |
| Eliza Rutherford, |       | Serkan    | Cabi,     | Tengda    | Han, Zhitao   | Gong, |                  |         |              |         |           |            |       |        |
|                   |       |           |           |           |               |       | Valentin         | Haenel, | Ingo         | Fruend, | Peter     | Yianilos,  |       | Moritz |
| Sina Samangooei,  |       | Marianne  | Monteiro, |           | Jacob Menick, | Se-   |                  |         |              |         |           |            |       |        |
|                   |       |           |           |           |               |       | Mueller-Freitag, |         | et al.       | The”    | something | something” |       | video  |
bastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sa- database for learning and evaluating visual common sense.
| hand Sharifzadeh, |     | Mikolaj | Binkowski, |     | Ricardo | Barreira, | InICCV,2017. |     | 1,5 |     |     |     |     |     |
| ----------------- | --- | ------- | ---------- | --- | ------- | --------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
| Oriol Vinyals,    |     | Andrew  | Zisserman, | and | Karen   | Simonyan. |              |     |     |     |     |     |     |     |
[19] AndrewJaegle,SebastianBorgeaud,Jean-BaptisteAlayrac,
| Flamingo: | a visual | language | model | for | few-shot | learning, |      |          |         |          |       |       |        |      |
| --------- | -------- | -------- | ----- | --- | -------- | --------- | ---- | -------- | ------- | -------- | ----- | ----- | ------ | ---- |
|           |          |          |       |     |          |           | Carl | Doersch, | Catalin | Ionescu, | David | Ding, | Skanda | Kop- |
| 2022. 1,3 |          |          |       |     |          |           |      |          |         |          |       |       |        |      |
pula,DanielZoran,AndrewBrock,EvanShelhamer,Olivier
[3] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen He´naff, Matthew M. Botvinick, Andrew Zisserman, Oriol
Sun,MarioLucˇic´,andCordeliaSchmid. Vivit: Avideovi- Vinyals, and Joa˜o Carreira. Perceiver io: A general archi-
| siontransformer. |     | InICCV,2021. |     | 1,2,3,5,6 |     |     |         |     |            |        |            |     |       |          |
| ---------------- | --- | ------------ | --- | --------- | --- | --- | ------- | --- | ---------- | ------ | ---------- | --- | ----- | -------- |
|                  |     |              |     |           |     |     | tecture | for | structured | inputs | & outputs. | In  | arXiv | preprint |
[4] Max Bain, Arsha Nagrani, Gu¨l Varol, and Andrew Zisser- arXiv:2107.14795,2021. 1,3
man. Frozenintime: Ajointvideoandimageencoderfor [20] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang,
end-to-endretrieval. InICCV,2021. 1,3 Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola,
|            |      |          |         |       |     |           | Tim | Green, | Trevor Back, |     | Paul Natsev, | et  | al. | The ki- |
| ---------- | ---- | -------- | ------- | ----- | --- | --------- | --- | ------ | ------------ | --- | ------------ | --- | --- | ------- |
| [5] Hangbo | Bao, | Li Dong, | Songhao | Piao, | and | Furu Wei. |     |        |              |     |              |     |     |         |
Beit: Bert pre-training of image transformers. In netics human action video dataset. In arXiv preprint
arXiv:https://arxiv.org/abs/2106.08254,2021. 1 arXiv:1705.06950,2017. 5
[6] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is [21] Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang,
|            |           |     |          |           |                |     | Mingxing |     | Tan, Matthew |     | Brown, | and | Boqing | Gong. |
| ---------- | --------- | --- | -------- | --------- | -------------- | --- | -------- | --- | ------------ | --- | ------ | --- | ------ | ----- |
| space-time | attention | all | you need | for video | understanding? |     |          |     |              |     |        |     |        |       |
InICML,2021. 1,2,3,5,6 Movinets: Mobilevideonetworksforefficientvideorecog-
[7] JoaoCarreira,EricNoland,ChloeHillier,andAndrewZis- nition. InCVPR,2021. 5,6,11
serman. A short note on the kinetics-700 human action [22] Youngwan Lee, Hyung-Il Kim, and Jinyoung Moon
|          |                                       |     |     |     |     |     | Kimin | Yun. | Diverse | temporal | aggregation |     | and depthwise |     |
| -------- | ------------------------------------- | --- | --- | --- | --- | --- | ----- | ---- | ------- | -------- | ----------- | --- | ------------- | --- |
| dataset. | InarXivpreprintarXiv:1907.06987,2019. |     |     |     |     | 5   |       |      |         |          |             |     |               |     |
spatiotemporalfactorizationforefficientvideoclassification.
| [8] Joao Carreira | and | Andrew | Zisserman. |     | Quo vadis, | action |     |     |     |     |     |     |     |     |
| ----------------- | --- | ------ | ---------- | --- | ---------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
recognition?anewmodelandthekineticsdataset.InCVPR, InarXivpreprintarXiv:2012.00317,2020. 6
2017. 1,2 [23] Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Man-
|     |     |     |     |     |     |     | galam, | Bo  | Xiong, Jitendra | Malik, | and | Christoph | Feichten- |     |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | --------------- | ------ | --- | --------- | --------- | --- |
[9] EkinD.Cubuk,BarretZoph,JonathonShlens,andQuocV.
|                  |     |           |           |     |                   |     | hofer. | Mvitv2: | Improvedmultiscalevisiontransformersfor |     |     |     |     |     |
| ---------------- | --- | --------- | --------- | --- | ----------------- | --- | ------ | ------- | --------------------------------------- | --- | --- | --- | --- | --- |
| Le. Randaugment: |     | Practical | automated |     | data augmentation |     |        |         |                                         |     |     |     |     |     |
classificationanddetection.InProceedingsoftheIEEE/CVF
| withareducedsearchspace. |     |         | InNeurIPS,2020. |              | 9,10         |     |                      |        |             |        |      |         |              |       |
| ------------------------ | --- | ------- | --------------- | ------------ | ------------ | --- | -------------------- | ------ | ----------- | ------ | ---- | ------- | ------------ | ----- |
|                          |     |         |                 |              |              |     | Conference           |        | on Computer | Vision | and  | Pattern | Recognition, |       |
| [10] Alexey Dosovitskiy, |     | Lucas   | Beyer,          | Alexander    | Kolesnikov,  |     |                      |        |             |        |      |         |              |       |
|                          |     |         |                 |              |              |     | pages4804–4814,2022. |        |             | 6      |      |         |              |       |
| Dirk Weissenborn,        |     | Xiaohua |                 | Zhai, Thomas | Unterthiner, |     |                      |        |             |        |      |         |              |       |
|                          |     |         |                 |              |              |     | [24] Youwei          | Liang, | Chongjian   | Ge,    | Zhan | Tong,   | Yibing       | Song, |
MostafaDehghani,MatthiasMinderer,GeorgHeigold,Syl-
|                                          |     |                                   |     |     |           |     | Jue    | Wang,                                           | and Pengtao | Xie. | Not all | patches | are what | you |
| ---------------------------------------- | --- | --------------------------------- | --- | --- | --------- | --- | ------ | ----------------------------------------------- | ----------- | ---- | ------- | ------- | -------- | --- |
| vainGelly,JakobUszkoreit,andNeilHoulsby. |     |                                   |     |     | Animageis |     |        |                                                 |             |      |         |         |          |     |
|                                          |     |                                   |     |     |           |     | need:  | Expeditingvisiontransformersviatokenreorganiza- |             |      |         |         |          |     |
| worth16x16words:                         |     | Transformersforimagerecognitionat |     |     |           |     |        |                                                 |             |      |         |         |          |     |
|                                          |     |                                   |     |     |           |     | tions. | arXivpreprintarXiv:2202.07800,2022.             |             |      |         |         | 2        |     |
| scale. InICLR,2021.                      |     | 1,3,9                             |     |     |           |     |        |                                                 |             |      |         |         |          |     |
[25] ValeriiLikhosherstov,AnuragArnab,KrzysztofChoroman-
[11] XianzhiDu,YeqingLi,YinCui,RuiQian,JingLi,andIrwan
|                                                |     |     |     |     |     |         | ski,    | Mario    | Lucic, Yi                              | Tay, Adrian | Weller, | and | Mostafa | De- |
| ---------------------------------------------- | --- | --- | --- | --- | --- | ------- | ------- | -------- | -------------------------------------- | ----------- | ------- | --- | ------- | --- |
| Bello. Revisiting3dresnetsforvideorecognition. |     |     |     |     |     | InarXiv |         |          |                                        |             |         |     |         |     |
|                                                |     |     |     |     |     |         | hghani. | Polyvit: | Co-trainingvisiontransformersonimages, |             |         |     |         |     |
preprintarXiv:2109.01696,2021. 5,6 videos and audio. arXiv preprint arXiv:2111.12993, 2021.
[12] HaodongDuan,YueZhao,YuanjunXiong,WentaoLiu,and
1,3,6
| Dahua Lin.        | Omni-sourced |              | webly-supervised |     | learning | for |                                       |        |      |          |      |              |          |       |
| ----------------- | ------------ | ------------ | ---------------- | --- | -------- | --- | ------------------------------------- | ------ | ---- | -------- | ---- | ------------ | -------- | ----- |
|                   |              |              |                  |     |          |     | [26] Ji Lin,                          | Chuang | Gan, | and Song | Han. | Tsm:         | Temporal | shift |
| videorecognition. |              | InECCV,2020. |                  | 2,5 |          |     |                                       |        |      |          |      |              |          |       |
|                   |              |              |                  |     |          |     | moduleforefficientvideounderstanding. |        |      |          |      | InICCV,2019. |          | 5     |
[13] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, [27] Ziyi Lin, Shijie Geng, Renrui Zhang, Peng Gao, Gerard
ZhichengYan,JitendraMalik,andChristophFeichtenhofer.
|                               |     |     |     |              |       |     | deMelo,  | XiaogangWang,                              |     | JifengDai, |     | YuQiao, | andHong- |     |
| ----------------------------- | --- | --- | --- | ------------ | ----- | --- | -------- | ------------------------------------------ | --- | ---------- | --- | ------- | -------- | --- |
| Multiscalevisiontransformers. |     |     |     | InICCV,2021. | 1,5,6 |     |          |                                            |     |            |     |         |          |     |
|                               |     |     |     |              |       |     | shengLi. | Frozenclipmodelsareefficientvideolearners. |     |            |     |         |          | In  |
[14] ChristophFeichtenhofer. X3d: Expandingarchitecturesfor EuropeanConferenceonComputerVision,pages388–404.
| efficientvideorecognition. |     |     | InCVPR,2020. |     | 1,5,6 |     | Springer,2022. |     | 1   |     |     |     |     |     |
| -------------------------- | --- | --- | ------------ | --- | ----- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
[15] ChristophFeichtenhofer,HaoqiFan,YanghaoLi,andKaim- [28] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang,
ing He. Masked autoencoders as spatiotemporal learners. StephenLin,andHanHu. Videoswintransformer. InarXiv
arXivpreprintarXiv:2205.09113,2022. 1,2,3,5 preprintarXiv:2106.13230,2021. 1

[29] Dmitrii Marin, Jen-Hao Rick Chang, Anurag Ranjan, standing. In European Conference on Computer Vision,
Anish Prabhu, Mohammad Rastegari, and Oncel Tuzel. pages510–526.Springer,2016. 9
Token pooling in vision transformers. arXiv preprint [43] KarenSimonyanandAndrewZisserman. Two-streamcon-
arXiv:2110.03860,2021. 2 volutional networks for action recognition in videos. In
[30] Muhammad Muzammal Naseer, Kanchana Ranasinghe, NeurIPS,2014. 2
SalmanHKhan,MunawarHayat,FahadShahbazKhan,and [44] Andreas Steiner, Alexander Kolesnikov, , Xiaohua Zhai,
Ming-HsuanYang.Intriguingpropertiesofvisiontransform- Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How
|               |     |           |             |     |            |          | totrainyourvit? |     | data, | augmentation, |     | andregularizationin |     |     |
| ------------- | --- | --------- | ----------- | --- | ---------- | -------- | --------------- | --- | ----- | ------------- | --- | ------------------- | --- | --- |
| ers. Advances |     | in Neural | Information |     | Processing | Systems, |                 |     |       |               |     |                     |     |     |
34:23296–23308,2021. 2 vision transformers. In arXiv preprint arXiv:2106.10270,
| [31] BolinNi,HouwenPeng,MinghaoChen,SongyangZhang, |     |     |     |     |     |     | 2021. | 6   |     |     |     |     |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
Gaofeng Meng, Jianlong Fu, Shiming Xiang, and Haibin [45] Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia
|     |     |     |     |     |     |     | Schmid. | Segmenter: |     | Transformer | for | semantic | segmenta- |     |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------- | --- | ----------- | --- | -------- | --------- | --- |
Ling.Expandinglanguage-imagepretrainedmodelsforgen-
|                       |     |     |              |     |     |     | tion. | InICCV,2021. |     | 1   |     |     |     |     |
| --------------------- | --- | --- | ------------ | --- | --- | --- | ----- | ------------ | --- | --- | --- | --- | --- | --- |
| eralvideorecognition. |     |     | InECCV,2022. |     | 1   |     |       |              |     |     |     |     |     |     |
[32] SeongHyeonPark,JihoonTack,ByeonghoHeo,Jung-Woo [46] ChenSun, AbhinavShrivastava, SaurabhSingh, andAbhi-
Ha,andJinwooShin.K-centeredpatchsamplingforefficient navGupta. Revisitingunreasonableeffectivenessofdatain
|                                    |              |     |          |            |     |             | deeplearningera. |       | InICCV,2017. |           | 2     |     |       |       |
| ---------------------------------- | ------------ | --- | -------- | ---------- | --- | ----------- | ---------------- | ----- | ------------ | --------- | ----- | --- | ----- | ----- |
| video                              | recognition. | In  | European | Conference |     | on Computer |                  |       |              |           |       |     |       |       |
|                                    |              |     |          |            |     |             | [47] Zhan        | Tong, | Yibing       | Song, Jue | Wang, | and | Limin | Wang. |
| Vision,pages160–176.Springer,2022. |              |     |          |            | 2,3 |             |                  |       |              |           |       |     |       |       |
[33] Mandela Patrick, Dylan Campbell, Yuki M Asano, Is- Videomae: Masked autoencoders are data-efficient learn-
han Misra Florian Metze, Christoph Feichtenhofer, Andrea ers for self-supervised video pre-training. arXiv preprint
|                           |     |     |     |                          |     |     | arXiv:2203.12602,2022. |     |     | 1,2,3,5,6 |     |     |     |     |
| ------------------------- | --- | --- | --- | ------------------------ | --- | --- | ---------------------- | --- | --- | --------- | --- | --- | --- | --- |
| Vedaldi,JoHenriques,etal. |     |     |     | Keepingyoureyeontheball: |     |     |                        |     |     |           |     |     |     |     |
[48] DuTran,LubomirBourdev,RobFergus,LorenzoTorresani,
Trajectoryattentioninvideotransformers.InNeurIPS,2021.
|     |     |     |     |     |     |     | andManoharPaluri. |     |     | Learningspatiotemporalfeatureswith |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | ---------------------------------- | --- | --- | --- | --- |
5,6
InICCV,2015.
[34] AJ Piergiovanni, Anelia Angelova, Alexander Toshev, and 3dconvolutionalnetworks. 1,2
|                                                 |     |                                       |     |              |     |               | [49] Du Tran,   | Heng                 | Wang,        | Lorenzo | Torresani,        |     | and Matt | Feis- |
| ----------------------------------------------- | --- | ------------------------------------- | --- | ------------ | --- | ------------- | --------------- | -------------------- | ------------ | ------- | ----------------- | --- | -------- | ----- |
| MichaelSRyoo.                                   |     | Evolvingspace-timeneuralarchitectures |     |              |     |               |                 |                      |              |         |                   |     |          |       |
|                                                 |     |                                       |     |              |     |               | zli.            | Video classification |              | with    | channel-separated |     | convolu- |       |
| for videos.                                     | In  | Proceedings                           | of  | the IEEE/CVF |     | International |                 |                      |              |         |                   |     |          |       |
|                                                 |     |                                       |     |              |     |               | tionalnetworks. |                      | InICCV,2019. |         | 5                 |     |          |       |
| ConferenceonComputerVision,pages1793–1802,2019. |     |                                       |     |              |     | 2             |                 |                      |              |         |                   |     |          |       |
[50] DuTran,HengWang,LorenzoTorresani,JamieRay,Yann
| [35] AJ Piergiovanni, |     | Kairo | Morton, | Weicheng | Kuo, | Michael |     |     |     |     |     |     |     |     |
| --------------------- | --- | ----- | ------- | -------- | ---- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
LeCun,andManoharPaluri.Acloserlookatspatiotemporal
| Ryoo,andAneliaAngelova.             |     |     |     | Videoquestionansweringwith |     |     |                                   |     |     |     |     |              |     |     |
| ----------------------------------- | --- | --- | --- | -------------------------- | --- | --- | --------------------------------- | --- | --- | --- | --- | ------------ | --- | --- |
|                                     |     |     |     |                            |     |     | convolutionsforactionrecognition. |     |     |     |     | InCVPR,2018. |     | 2   |
| iterativevideo-textco-tokenization. |     |     |     | ECCV,2022.                 |     | 2   |                                   |     |     |     |     |              |     |     |
[51] AshishVaswani,NoamShazeer,NikiParmar,JakobUszko-
[36] ZhaofanQiu,TingYao,Chong-WahNgo,XinmeiTian,and
reit,LlionJones,AidanNGomez,ŁukaszKaiser,andIllia
TaoMei.Learningspatio-temporalrepresentationwithlocal
|                     |     |              |     |     |     |     | Polosukhin. | Attentionisallyouneed. |     |     |     | InNeurIPS,2017. |     | 2,  |
| ------------------- | --- | ------------ | --- | --- | --- | --- | ----------- | ---------------------- | --- | --- | --- | --------------- | --- | --- |
| andglobaldiffusion. |     | InCVPR,2019. |     | 5   |     |     |             |                        |     |     |     |                 |     |     |
3,4
| [37] Yongming                                        | Rao,        | Wenliang     | Zhao, | Benlin          | Liu, Jiwen | Lu, Jie  |                                     |               |     |                |                       |                 |      |        |
| ---------------------------------------------------- | ----------- | ------------ | ----- | --------------- | ---------- | -------- | ----------------------------------- | ------------- | --- | -------------- | --------------------- | --------------- | ---- | ------ |
|                                                      |             |              |       |                 |            |          | [52] HengWangandCordeliaSchmid.     |               |     |                | Actionrecognitionwith |                 |      |        |
| Zhou,                                                | and Cho-Jui | Hsieh.       |       | Dynamicvit:     | Efficient  | vision   |                                     |               |     |                |                       |                 |      |        |
|                                                      |             |              |       |                 |            |          | improved                            | trajectories. |     | In Proceedings |                       | of the          | IEEE | inter- |
| transformers                                         |             | with dynamic | token | sparsification. |            | Advances |                                     |               |     |                |                       |                 |      |        |
|                                                      |             |              |       |                 |            |          | nationalconferenceoncomputervision, |               |     |                |                       | pages3551–3558, |      |        |
| inneuralinformationprocessingsystems,34:13937–13949, |             |              |       |                 |            |          | 2013.                               | 2             |     |                |                       |                 |      |        |
| 2021.                                                | 2           |              |       |                 |            |          |                                     |               |     |                |                       |                 |      |        |
[53] JunkeWang,DongdongChen,ZuxuanWu,ChongLuo,Lu-
[38] MichaelS.Ryoo,AJPiergiovanni,AnuragArnab,Mostafa
|           |     |        |           |               |     |          | owei   | Zhou,  | Yucheng | Zhao,   | Yujia | Xie, Ce    | Liu, Yu-Gang |     |
| --------- | --- | ------ | --------- | ------------- | --- | -------- | ------ | ------ | ------- | ------- | ----- | ---------- | ------------ | --- |
| Dehghani, | and | Anelia | Angelova. | Tokenlearner: |     | Adaptive |        |        |         |         |       |            |              |     |
|           |     |        |           |               |     |          | Jiang, | and Lu | Yuan.   | Omnivl: | One   | foundation | model        | for |
space-timetokenizationforvideos. 2021. 2,3,5,6,9,11 image-languageandvideo-languagetasks. 2022. 1,3
[39] MichaelSRyoo,AJPiergiovanni,JuhanaKangaspunta,and
|                                         |           |                |     |            |                |          | [54] Junke       | Wang,            | Xitong | Yang,                          | Hengduo | Li, Li   | Liu, Zuxuan |     |
| --------------------------------------- | --------- | -------------- | --- | ---------- | -------------- | -------- | ---------------- | ---------------- | ------ | ------------------------------ | ------- | -------- | ----------- | --- |
| Anelia                                  | Angelova. | Assemblenet++: |     | Assembling |                | modality |                  |                  |        |                                |         |          |             |     |
|                                         |           |                |     |            |                |          | Wu,              | andYu-GangJiang. |        | Efficientvideotransformerswith |         |          |             |     |
| representationsviaattentionconnections. |           |                |     |            | InEuropeanCon- |          |                  |                  |        |                                |         |          |             |     |
|                                         |           |                |     |            |                |          | spatial-temporal |                  | token  | selection.                     | In      | European | Conference  |     |
ferenceonComputerVision,pages654–671.Springer,2020. onComputerVision,pages69–86.Springer,2022. 2
1,9,11 [55] Wenhai Wang, Enze Xie, Xiang Li, Kaitao Song Deng-
[40] MichaelSRyoo,AJPiergiovanni,MingxingTan,andAnelia
|           |              |     |                                |     |     |     | PingFan,                  | DingLiang, |     | TongLu,                    | PingLuo, |     | andLingShao. |     |
| --------- | ------------ | --- | ------------------------------ | --- | --- | --- | ------------------------- | ---------- | --- | -------------------------- | -------- | --- | ------------ | --- |
| Angelova. | Assemblenet: |     | Searchingformulti-streamneural |     |     |     |                           |            |     |                            |          |     |              |     |
|           |              |     |                                |     |     |     | Pyramidvisiontransformer: |            |     | Aversatilebackbonefordense |          |     |              |     |
connectivityinvideoarchitectures. InICLR,2019. 11 predictionwithoutconvolutions. InICCV,2021. 1
[41] WenzheShi,JoseCaballero,FerencHusza´r,JohannesTotz, [56] XiaolongWang,RossGirshick,AbhinavGupta,andKaim-
AndrewPAitken,RobBishop,DanielRueckert,andZehan ingHe. Non-localneuralnetworks. InCVPR,2018. 1,5
| Wang. | Real-time | single | image | and video | super-resolution |     |                 |     |         |                      |     |     |          |     |
| ----- | --------- | ------ | ----- | --------- | ---------------- | --- | --------------- | --- | ------- | -------------------- | --- | --- | -------- | --- |
|       |           |        |       |           |                  |     | [57] YufeiWang, |     | DuTran, | andLorenzoTorresani. |     |     | Unidual: | A   |
usinganefficientsub-pixelconvolutionalneuralnetwork. In unifiedmodelforimageandvideounderstanding. InarXiv
ProceedingsoftheIEEEconferenceoncomputervisionand preprintarXiv:1906.03857,2019. 2,3
patternrecognition,pages1874–1883,2016. 4 [58] Yu-An Wang and Yun-Nung Chen. What do position
[42] Gunnar A Sigurdsson, Gu¨l Varol, Xiaolong Wang, Ali embeddings learn? an empirical study of pre-trained
Farhadi, Ivan Laptev, and Abhinav Gupta. Hollywood in language model positional encoding. arXiv preprint
homes: Crowdsourcing data collection for activity under- arXiv:2010.04903,2022. 3

[59] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu,
Alan Yuille, and Christoph Feichtenhofer. Masked fea-
ture prediction for self-supervised visual pre-training. In
arXiv:https://arxiv.org/abs/2112.09133,2021. 6
[60] ZuxuanWu,CaimingXiong,Chih-YaoMa,RichardSocher,
andLarrySDavis. Adaframe: Adaptiveframeselectionfor
fast video recognition. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
pages1278–1287,2019. 2
[61] SainingXie,ChenSun,JonathanHuang,ZhuowenTu,and
KevinMurphy. Rethinkingspatiotemporalfeaturelearning:
Speed-accuracytrade-offsinvideoclassification. InECCV,
2018. 1
[62] Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi
Zhang, Chen Sun, and Cordelia Schmid. Multiview trans-
formersforvideorecognition. InCVPR,2022. 1,2,3,5,6,
7,9
[63] LeweiYao,RunhuiHuang,LuHou,GuansongLu,Minzhe
Niu,HangXu,XiaodanLiang,ZhenguoLi,XinJiang,and
ChunjingXu.Filip:Fine-grainedinteractivelanguage-image
pre-training. ICLR,2022. 1
[64] JiahuiYu,ZiruiWang,VijayVasudevan,LeggYeung,Mo-
jtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive
captionersareimage-textfoundationmodels,2022. 5,6
[65] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella,
Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang,
Boxin Li, Chunyuan Li, et al. Florence: A new foun-
dation model for computer vision. In arXiv preprint
arXiv:2111.11432,2021. 5,6
[66] Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yan-
peng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack
Hessel,AliFarhadi,andYejinChoi. Merlotreserve:Neural
scriptknowledgethroughvisionandlanguageandsound. In
ProceedingsoftheIEEE/CVFConferenceonComputerVi-
sionandPatternRecognition,pages16375–16387,2022. 1,
6
[67] Xiaohua Zhai, Alexander Kolesnikov, and Lucas Beyer
NeilHoulsby. Scalingvisiontransformers. InCVPR,2022.
1
[68] Bowen Zhang, Jiahui Yu, Christopher Fifty, Wei Han, An-
drewMDai,RuomingPang,andFeiSha. Co-trainingtrans-
formerwithvideosandimagesimprovesactionrecognition.
InarXivpreprintarXiv:2112.07175,2021. 1,3,5,6
[69] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and
DavidLopez-Paz. Mixup:Beyondempiricalriskminimiza-
tion. InICLR,2018. 10
[70] YanyiZhang, XinyuLi, ChunhuiLiu, BingShuai, YiZhu,
BiagioBrattoli, HaoChen, IvanMarsic, andJosephTighe.
Vidtr: Video transformer without convolutions. In ICCV,
2021. 5,6
