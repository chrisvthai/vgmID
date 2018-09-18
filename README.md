# Video Game Music Genre Identifier
Quite a lengthy name, but basically a neural network aimed at identifying the video game genres a game soundtrack belongs to.
You give it a .wav file, and the network will try to guess what genre the soundtrack is from.

To run the program on a .wav file (it's imperative it be 32-bit float PCM format! sci-kit won't let me do otherwise), run the command:

```python vgm_id.py -predict_wav=song_name```

Replace 'song_name' with the filename of the song you're testing with.


## Structure
To morph the input .wav file into input for the neural network, I read it in and produce a spectrogram using the *scipy* library.
Next, I downsample the immensely large matrix from this process into a much more manageable size for the model to handle. From then on,
it's just a typical convolutional neural network. 

## Training
To train the network, I gathered a bunch of MP3 files from SoundCloud of some video game soundtracks, both notable and not so notable.
I then used Audacity to convert these into mono-channel WAV files before putting them into one big folder, organized as such:

/songs  
&nbsp;&nbsp;&nbsp;&nbsp;/Adventure  
&nbsp;&nbsp;&nbsp;&nbsp;/Casual  
&nbsp;&nbsp;&nbsp;&nbsp;/Fighting  
&nbsp;&nbsp;&nbsp;&nbsp;/Horror  
&nbsp;&nbsp;&nbsp;&nbsp;/RPG  
&nbsp;&nbsp;&nbsp;&nbsp;/Shooter  
&nbsp;&nbsp;&nbsp;&nbsp;/Strategy

These classifications are highly subjective, and generally it isn't wise to try to lump a video game into a single category, for many
reasons. Most notably, many games blend elements of different genres; for instance, *Bioshock* combines RPG, shooter, and horror elements.
There can also be games withing a specific 'genre' (think Platforming games, or to an extent Action games) that don't really do much to
differentiate the game stylistically. For reasons like this, I did not think it wise to add a 'Platforming' genre, for instance. I could
foresee some confusion between lumping *Mario* and *Cuphead* in the same platformer category.

In addition, in the same vein as the previous statement, stylistically different games (and such, wholly different styles of music. Think
Fantasy RPG versus SciFi RPG) would have to be accounted for. In summary, do not think of these categories as definitive. Rather, think of
them as merely something I wanted to experiment with and see if a neural network could maybe learn these patterns.

One more thing: there are so many video game soundtracks across the entire history of gaming that it would take me forever to gather
a set of training files completely representative all games. As such, there could possibly be many different patterns my current network
won't be able to recognize.

Anyways, to train the network, have a ```/songs``` folder such as above, and run the command 

```python vgm-id.py --train```
