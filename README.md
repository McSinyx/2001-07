# Speaker Recognition Experiments

> What's the difference between me and you?    
> You talk a good one, but you don't do what you supposed to do


## Experiments

[GitHub topic] is a good start for exploration.

### naive

    usage: python -m naive.train [-h] source dest
    usage: python -m naive.test [-h] models data
    usage: python -m naive.run [-h] [-l] [-d DEVICE] models

This approach extract [Mel-frequency cepstrum] from audios and try to score
the sample directly, as use by [PyGender-Voice][gender blog].  The original
data set with compatible layout can be found [here][gender data].


## Data

### Paragraphs

1. The doctor is a person who looks after the sick people and prescribes
   medicines so that the patient recovers fast.  In order to become a doctor,
   a person has to study medicine.  Doctors lead a hard life.  Their life
   is very busy. They get up early in the morning and go to the hospital.
   They work without taking a break.  They always remain polite so that
   patients feel comfortable with them.  Since doctors work so hard we must
   realise their value.
2. Tea is an aromatic beverage commonly prepared by pouring hot or boiling
   water over cured leaves of an evergreen shrub native to East Asia.
   After water, it is the most widely consumed drink in the world.  There are
   many different types of tea; some, like Darjeeling and Chinese greens,
   have a cooling, slightly bitter, and astringent flavour, while others have
   vastly different profiles that include sweet, nutty, floral or grassy notes.
3. Flowers are the reproductive parts of flowering plants.  A flower is
   a special part of the plant.  Flowers are also called the bloom or blossom
   of a plant.  Flowers have petals.  Inside the part of the flower that has
   petals are the parts which produce pollen and seeds.  Flowers may grow
   separately on the plant, or they may grow together in an inflorescence.

### Small Talk

* Morning, Paul.
* Oh, morning, William, how are you?
* Fine, thanks.  Have a good weekend.
* Yes, thanks.  Catch you later.
* OK, see you.

Note: this one should be numbered as 4.

### Dialogues

Daily life [dialogues] number 5, 6, 8, 9 and 10.


## Tests

### Dialogues

Daily life [dialogues] number 1, 2, 3, 4 and 7.

[gender blog]: https://appliedmachinelearning.wordpress.com/2017/06/14/voice-gender-detection-using-gmms-a-python-primer/
[gender data]: https://drive.google.com/open?id=1u28uP2TlL0c-VKNj3eEw9-NdsOhx-XXq
[GitHub topic]: https://github.com/topics/speaker-recognition
[Mel-frequency cepstrum]: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
[dialogues]: https://www.eslfast.com/easydialogs/dailylife.htm
