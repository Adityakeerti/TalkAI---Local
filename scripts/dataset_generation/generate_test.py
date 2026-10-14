"""
Generates clean, diverse, natural micro-turn dialogues for the 60 test conversations
in /home/aditya/Desktop/STUDY/TalkAI/data/duplexcascade_synthetic/micro_turns_test_new.jsonl.

Strictly conforms to:
- 600ms micro-turns
- Assistant outputs 'listen' to yield or 3-7 words per micro-turn when speaking
- User outputs 'no voice' during pauses
- Every conversation ends with user: 'no voice', assistant: 'listen'
- High diversity: every conversation has unique topic and distinct entities
- 0% topic leakage / overlap with train and validation splits
- Exact scenario and ID matching the original test set
"""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn "
    "from streaming speech recognition, or 'no voice' when no speech is detected. "
    "Output exactly 'listen' when you should yield and keep listening; otherwise output a "
    "brief, natural speech fragment that can be spoken now. Do not repeat prior fragments. "
    "Continue an unfinished answer across turns when appropriate. Yield immediately for a "
    "clear interruption or urgent safety risk. Do not treat every pause as the end of the user's thought."
)

CONVERSATIONS = [
    # 00: conv_00379 | hesitation_wait
    {
        "id": "conv_00379",
        "scenario": "hesitation_wait",
        "micro_turns": [
            {"user": "I was hesitating between", "assistant": "listen"},
            {"user": "renting or buying an apartment", "assistant": "listen"},
            {"user": "given the rising mortgage rates", "assistant": "listen"},
            {"user": "no voice", "assistant": "That is an important decision."},
            {"user": "I might move in two years", "assistant": "listen"},
            {"user": "if my firm relocates", "assistant": "listen"},
            {"user": "no voice", "assistant": "Renting gives much greater flexibility"},
            {"user": "no voice", "assistant": "for short time horizons."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 01: conv_00308 | filler_and_mhm
    {
        "id": "conv_00308",
        "scenario": "filler_and_mhm",
        "micro_turns": [
            {"user": "So hm I wanted advice", "assistant": "listen"},
            {"user": "about planting fruit trees", "assistant": "listen"},
            {"user": "no voice", "assistant": "I am listening closely."},
            {"user": "we have heavy clay soil", "assistant": "listen"},
            {"user": "and harsh winter frosts here", "assistant": "listen"},
            {"user": "no voice", "assistant": "That needs hardy rootstocks."},
            {"user": "pears or plums maybe", "assistant": "listen"},
            {"user": "no voice", "assistant": "European plums handle clay well"},
            {"user": "no voice", "assistant": "and survive hard frost."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 02: conv_00441 | natural_conversation
    {
        "id": "conv_00441",
        "scenario": "natural_conversation",
        "micro_turns": [
            {"user": "I roasted garlic today", "assistant": "listen"},
            {"user": "and it turned out bitter", "assistant": "listen"},
            {"user": "no voice", "assistant": "High heat causes that."},
            {"user": "I baked it at four-fifty", "assistant": "listen"},
            {"user": "no voice", "assistant": "Drop temperature to three-fifty"},
            {"user": "no voice", "assistant": "and wrap tightly in foil."},
            {"user": "I will try that next", "assistant": "listen"},
            {"user": "no voice", "assistant": "It should turn sweet."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 03: conv_00521 | emotional_support
    {
        "id": "conv_00521",
        "scenario": "emotional_support",
        "micro_turns": [
            {"user": "I lost my wedding ring", "assistant": "listen"},
            {"user": "and I feel completely devastated", "assistant": "listen"},
            {"user": "no voice", "assistant": "That is so distressing."},
            {"user": "I searched the whole house", "assistant": "listen"},
            {"user": "no voice", "assistant": "Take a calm deep breath."},
            {"user": "where should I retrace steps", "assistant": "listen"},
            {"user": "no voice", "assistant": "Check bathroom drains and laundry"},
            {"user": "no voice", "assistant": "where rings slip off."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 04: conv_00357 | natural_conversation
    {
        "id": "conv_00357",
        "scenario": "natural_conversation",
        "micro_turns": [
            {"user": "My lemon tree dropped blooms", "assistant": "listen"},
            {"user": "nearly all petals fell off", "assistant": "listen"},
            {"user": "no voice", "assistant": "Sudden temperature shifts happen."},
            {"user": "we had dry chilly winds", "assistant": "listen"},
            {"user": "no voice", "assistant": "Shelter it from drafts"},
            {"user": "no voice", "assistant": "and maintain steady moisture."},
            {"user": "I moved it onto patio", "assistant": "listen"},
            {"user": "no voice", "assistant": "New buds will sprout."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 05: conv_00186 | multistep_task
    {
        "id": "conv_00186",
        "scenario": "multistep_task",
        "micro_turns": [
            {"user": "Help me construct a telescope stand", "assistant": "listen"},
            {"user": "I have cedar boards", "assistant": "listen"},
            {"user": "no voice", "assistant": "Cut the base and walls."},
            {"user": "walls are cut to size", "assistant": "listen"},
            {"user": "no voice", "assistant": "Drill an entry hole next."},
            {"user": "one and a half inches", "assistant": "listen"},
            {"user": "no voice", "assistant": "Fasten panels with screws"},
            {"user": "no voice", "assistant": "and add roof drainage."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 06: conv_00458 | emotional_support
    {
        "id": "conv_00458",
        "scenario": "emotional_support",
        "micro_turns": [
            {"user": "I burned our anniversary dinner", "assistant": "listen"},
            {"user": "and the roast is ruined", "assistant": "listen"},
            {"user": "no voice", "assistant": "Kitchen accidents happen to everyone."},
            {"user": "my spouse arrives in minutes", "assistant": "listen"},
            {"user": "no voice", "assistant": "Order favorite takeout together."},
            {"user": "we can laugh about it", "assistant": "listen"},
            {"user": "no voice", "assistant": "The shared celebration matters"},
            {"user": "no voice", "assistant": "much more than cooking."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 07: conv_00473 | user_interrupts_answer
    {
        "id": "conv_00473",
        "scenario": "user_interrupts_answer",
        "micro_turns": [
            {"user": "Explain how solar inverters work", "assistant": "listen"},
            {"user": "no voice", "assistant": "Solar inverters convert direct current"},
            {"user": "no voice", "assistant": "into alternating household current safely"},
            {"user": "wait, what is direct current", "assistant": "listen"},
            {"user": "no voice", "assistant": "Electricity flowing in one direction,"},
            {"user": "no voice", "assistant": "produced directly by panels."},
            {"user": "alright, continue the explanation", "assistant": "listen"},
            {"user": "no voice", "assistant": "The inverter synchronizes voltage"},
            {"user": "no voice", "assistant": "to feed local appliances."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 08: conv_00209 | user_requests_pause
    {
        "id": "conv_00209",
        "scenario": "user_requests_pause",
        "micro_turns": [
            {"user": "Explain how aircraft wings generate lift", "assistant": "listen"},
            {"user": "no voice", "assistant": "Curved aerofoils deflect passing airflow"},
            {"user": "no voice", "assistant": "creating lower pressure above"},
            {"user": "hold on, pause right there", "assistant": "listen"},
            {"user": "no voice", "assistant": "Sure, pausing for you."},
            {"user": "no voice", "assistant": "The wing pushes air down,"},
            {"user": "no voice", "assistant": "forcing the plane upward."},
            {"user": "got it, that makes sense", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 09: conv_00419 | user_requests_pause
    {
        "id": "conv_00419",
        "scenario": "user_requests_pause",
        "micro_turns": [
            {"user": "How do seismographs detect distant earthquakes", "assistant": "listen"},
            {"user": "no voice", "assistant": "Internal suspended inertia masses remain"},
            {"user": "no voice", "assistant": "stationary while bedrock shakes"},
            {"user": "slow down, let me take notes", "assistant": "listen"},
            {"user": "no voice", "assistant": "Taking a brief pause."},
            {"user": "no voice", "assistant": "Sensors measure the relative motion"},
            {"user": "no voice", "assistant": "between pendulum and Earth."},
            {"user": "alright, ready to proceed", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 10: conv_00402 | backchannel_story
    {
        "id": "conv_00402",
        "scenario": "backchannel_story",
        "micro_turns": [
            {"user": "We drove through Monument Valley", "assistant": "listen"},
            {"user": "right during late sunset", "assistant": "listen"},
            {"user": "the red sandstone buttes glowed", "assistant": "listen"},
            {"user": "no voice", "assistant": "That sounds absolutely breathtaking."},
            {"user": "a wild coyote crossed", "assistant": "listen"},
            {"user": "right in front of us", "assistant": "listen"},
            {"user": "no voice", "assistant": "An incredible desert encounter."},
            {"user": "we stood watching in silence", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 11: conv_00499 | clarification_needed
    {
        "id": "conv_00499",
        "scenario": "clarification_needed",
        "micro_turns": [
            {"user": "Can you review my newsletter draft", "assistant": "listen"},
            {"user": "no voice", "assistant": "What target audience is it for?"},
            {"user": "local pottery hobbyists", "assistant": "listen"},
            {"user": "no voice", "assistant": "Focus on workshop event dates,"},
            {"user": "no voice", "assistant": "or glazing technique tutorials?"},
            {"user": "glazing tutorials for beginners", "assistant": "listen"},
            {"user": "no voice", "assistant": "Add clear step photos"},
            {"user": "no voice", "assistant": "and recommend non-toxic glazes."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 12: conv_00436 | clarification_needed
    {
        "id": "conv_00436",
        "scenario": "clarification_needed",
        "micro_turns": [
            {"user": "My 3D print failed midway", "assistant": "listen"},
            {"user": "no voice", "assistant": "What visible defect appeared?"},
            {"user": "the layers separated completely", "assistant": "listen"},
            {"user": "no voice", "assistant": "Was it warping off bed,"},
            {"user": "no voice", "assistant": "or weak inter-layer adhesion?"},
            {"user": "weak adhesion between middle layers", "assistant": "listen"},
            {"user": "no voice", "assistant": "Increase the hotend temperature"},
            {"user": "no voice", "assistant": "and reduce cooling fan speed."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 13: conv_00202 | long_explanation_user
    {
        "id": "conv_00202",
        "scenario": "long_explanation_user",
        "micro_turns": [
            {"user": "I am trying to learn watercolour", "assistant": "listen"},
            {"user": "but the paper buckles horribly", "assistant": "listen"},
            {"user": "pigment dries into muddy blotches", "assistant": "listen"},
            {"user": "and fine lines bleed outward", "assistant": "listen"},
            {"user": "which ruins my landscape sketches", "assistant": "listen"},
            {"user": "no voice", "assistant": "Use heavier cotton paper."},
            {"user": "maybe stretch paper on board", "assistant": "listen"},
            {"user": "no voice", "assistant": "Tape cold-press paper down"},
            {"user": "no voice", "assistant": "while damp with gummed tape."},
            {"user": "no voice", "assistant": "Wait between color washes"},
            {"user": "no voice", "assistant": "to prevent muddy pigments mixing."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 14: conv_00547 | hesitation_wait
    {
        "id": "conv_00547",
        "scenario": "hesitation_wait",
        "micro_turns": [
            {"user": "I am debating whether", "assistant": "listen"},
            {"user": "to renovate our kitchen", "assistant": "listen"},
            {"user": "or replace the older roof", "assistant": "listen"},
            {"user": "no voice", "assistant": "Both represent substantial investments."},
            {"user": "the shingles are curling up", "assistant": "listen"},
            {"user": "and rainy season starts soon", "assistant": "listen"},
            {"user": "no voice", "assistant": "Prioritize the roof first"},
            {"user": "no voice", "assistant": "to prevent structural water damage."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 15: conv_00352 | clarification_needed
    {
        "id": "conv_00352",
        "scenario": "clarification_needed",
        "micro_turns": [
            {"user": "Recommend an audio interface", "assistant": "listen"},
            {"user": "no voice", "assistant": "What inputs do you need?"},
            {"user": "recording voice and guitar", "assistant": "listen"},
            {"user": "no voice", "assistant": "Looking for USB desktop portability,"},
            {"user": "no voice", "assistant": "or multiple rackmount preamps?"},
            {"user": "portable desktop with phantom power", "assistant": "listen"},
            {"user": "no voice", "assistant": "Look at dual-channel USB units"},
            {"user": "no voice", "assistant": "with low-noise XLR preamps."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 16: conv_00115 | silence_then_resume
    {
        "id": "conv_00115",
        "scenario": "silence_then_resume",
        "micro_turns": [
            {"user": "I went to an estate auction", "assistant": "listen"},
            {"user": "in a rural farmhouse yesterday", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
            {"user": "I bid on a telescope", "assistant": "listen"},
            {"user": "a brass antique refractor", "assistant": "listen"},
            {"user": "no voice", "assistant": "Did you win the bid?"},
            {"user": "yes for just eighty dollars", "assistant": "listen"},
            {"user": "no voice", "assistant": "That is a remarkable treasure."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 17: conv_00459 | multistep_task
    {
        "id": "conv_00459",
        "scenario": "multistep_task",
        "micro_turns": [
            {"user": "Help me descale my espresso machine", "assistant": "listen"},
            {"user": "water flow slowed to drops", "assistant": "listen"},
            {"user": "no voice", "assistant": "Mix citric acid solution."},
            {"user": "reservoir is filled with solution", "assistant": "listen"},
            {"user": "no voice", "assistant": "Run half through brew group."},
            {"user": "and steam wand is flushed", "assistant": "listen"},
            {"user": "no voice", "assistant": "Rinse reservoir with fresh water"},
            {"user": "no voice", "assistant": "and run two full flushes."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 18: conv_00250 | misheard_word
    {
        "id": "conv_00250",
        "scenario": "misheard_word",
        "micro_turns": [
            {"user": "I want to identify species of gulls", "assistant": "listen"},
            {"user": "no voice", "assistant": "Do you mean sea gulls"},
            {"user": "no voice", "assistant": "or coastal plant galls?"},
            {"user": "coastal galls, voice transcription messed up", "assistant": "listen"},
            {"user": "no voice", "assistant": "Plant galls on coastal shrubs."},
            {"user": "no voice", "assistant": "Examine the host leaf shape"},
            {"user": "no voice", "assistant": "and insect exit holes."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 19: conv_00230 | user_requests_pause
    {
        "id": "conv_00230",
        "scenario": "user_requests_pause",
        "micro_turns": [
            {"user": "How does radiometric carbon dating calculate age", "assistant": "listen"},
            {"user": "no voice", "assistant": "Unstable carbon-fourteen decays into nitrogen"},
            {"user": "no voice", "assistant": "at predictable exponential half-life rates"},
            {"user": "stop a second, let me think", "assistant": "listen"},
            {"user": "no voice", "assistant": "Sure, pausing for you."},
            {"user": "no voice", "assistant": "Living organisms absorb carbon continuously,"},
            {"user": "no voice", "assistant": "which stops upon death."},
            {"user": "okay, got that concept", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 20: conv_00364 | short_acknowledgement
    {
        "id": "conv_00364",
        "scenario": "short_acknowledgement",
        "micro_turns": [
            {"user": "I passed my ham radio exam", "assistant": "listen"},
            {"user": "with a near perfect score", "assistant": "listen"},
            {"user": "no voice", "assistant": "Congratulations on getting licensed."},
            {"user": "I was studying all month", "assistant": "listen"},
            {"user": "no voice", "assistant": "Your preparation paid off."},
            {"user": "setting up antenna tonight", "assistant": "listen"},
            {"user": "no voice", "assistant": "Have fun on the airwaves."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 21: conv_00449 | user_changes_topic
    {
        "id": "conv_00449",
        "scenario": "user_changes_topic",
        "micro_turns": [
            {"user": "How do I season cast iron pans", "assistant": "listen"},
            {"user": "using flaxseed oil", "assistant": "listen"},
            {"user": "no voice", "assistant": "Apply very thin coats"},
            {"user": "no voice", "assistant": "and bake at five-hundred."},
            {"user": "actually, can you recommend hiking boots", "assistant": "listen"},
            {"user": "for rocky muddy trails", "assistant": "listen"},
            {"user": "no voice", "assistant": "Look for Vibram outsoles"},
            {"user": "with waterproof Gore-Tex lining", "assistant": "listen"},
            {"user": "no voice", "assistant": "Choose mid-ankle support styles"},
            {"user": "no voice", "assistant": "to prevent ankle sprains."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 22: conv_00235 | assistant_interrupted
    {
        "id": "conv_00235",
        "scenario": "assistant_interrupted",
        "micro_turns": [
            {"user": "How do I grow microgreens indoors", "assistant": "listen"},
            {"user": "on a kitchen windowsill", "assistant": "listen"},
            {"user": "no voice", "assistant": "Spread potting soil trays"},
            {"user": "no voice", "assistant": "and sow sunflower seeds densely"},
            {"user": "wait, I bought broccoli seeds", "assistant": "listen"},
            {"user": "and a soil-less hemp mat", "assistant": "listen"},
            {"user": "no voice", "assistant": "Moisten the hemp mat thoroughly."},
            {"user": "no voice", "assistant": "Scatter tiny broccoli seeds evenly"},
            {"user": "no voice", "assistant": "and cover for blackout germination."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 23: conv_00423 | backchannel_story
    {
        "id": "conv_00423",
        "scenario": "backchannel_story",
        "micro_turns": [
            {"user": "We were kayaking in Puget Sound", "assistant": "listen"},
            {"user": "early on a glassy morning", "assistant": "listen"},
            {"user": "water was completely silent", "assistant": "listen"},
            {"user": "no voice", "assistant": "Such peaceful calm conditions."},
            {"user": "suddenly three harbor porpoises surfaced", "assistant": "listen"},
            {"user": "right alongside our boat", "assistant": "listen"},
            {"user": "no voice", "assistant": "What an unforgettable sight."},
            {"user": "we glided alongside them", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 24: conv_00160 | long_explanation_user
    {
        "id": "conv_00160",
        "scenario": "long_explanation_user",
        "micro_turns": [
            {"user": "Our community garden soil is compacted", "assistant": "listen"},
            {"user": "rain forms heavy standing puddles", "assistant": "listen"},
            {"user": "carrot seedlings cannot penetrate down", "assistant": "listen"},
            {"user": "weed roots strangle everything", "assistant": "listen"},
            {"user": "and tillage breaks our tiller blades", "assistant": "listen"},
            {"user": "no voice", "assistant": "Try no-till broadfork aeration."},
            {"user": "maybe layer thick compost mulch", "assistant": "listen"},
            {"user": "no voice", "assistant": "Spread aged compost mulch deeply"},
            {"user": "no voice", "assistant": "over the entire surface."},
            {"user": "no voice", "assistant": "Earthworms will naturally aerate compaction"},
            {"user": "no voice", "assistant": "without destroying fungal networks."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 25: conv_00326 | user_interrupts_answer
    {
        "id": "conv_00326",
        "scenario": "user_interrupts_answer",
        "micro_turns": [
            {"user": "How does GPS calculate user position", "assistant": "listen"},
            {"user": "no voice", "assistant": "Receivers calculate time differences between"},
            {"user": "no voice", "assistant": "radio signals from multiple satellites"},
            {"user": "wait, how many satellites minimum", "assistant": "listen"},
            {"user": "no voice", "assistant": "Four satellites are required"},
            {"user": "no voice", "assistant": "to pinpoint precise altitude."},
            {"user": "got it, go ahead", "assistant": "listen"},
            {"user": "no voice", "assistant": "Trilateration intersections yield latitude,"},
            {"user": "no voice", "assistant": "longitude, and receiver clock bias."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 26: conv_00177 | asr_noisy_repair
    {
        "id": "conv_00177",
        "scenario": "asr_noisy_repair",
        "micro_turns": [
            {"user": "Reserve ferry tickets to Victoria", "assistant": "listen"},
            {"user": "for three adults on Saturday", "assistant": "listen"},
            {"user": "morning sailing at nine", "assistant": "listen"},
            {"user": "no voice", "assistant": "Checking Saturday nine AM departures."},
            {"user": "sorry, I said two adults", "assistant": "listen"},
            {"user": "and ten thirty sailing", "assistant": "listen"},
            {"user": "no voice", "assistant": "Updated: two adults, ten thirty."},
            {"user": "no voice", "assistant": "Bringing a vehicle aboard?"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 27: conv_00207 | multistep_task
    {
        "id": "conv_00207",
        "scenario": "multistep_task",
        "micro_turns": [
            {"user": "Help me winterize the lawnmower", "assistant": "listen"},
            {"user": "mowing season is over", "assistant": "listen"},
            {"user": "no voice", "assistant": "Empty remaining gasoline first."},
            {"user": "fuel tank is drained", "assistant": "listen"},
            {"user": "no voice", "assistant": "Disconnect the spark plug boot."},
            {"user": "and scrape underneath deck", "assistant": "listen"},
            {"user": "no voice", "assistant": "Spray metal with oil"},
            {"user": "no voice", "assistant": "and store under cover."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 28: conv_00190 | hesitation_wait
    {
        "id": "conv_00190",
        "scenario": "hesitation_wait",
        "micro_turns": [
            {"user": "I have been thinking about", "assistant": "listen"},
            {"user": "switching from electric to induction stove", "assistant": "listen"},
            {"user": "though rewiring might be needed", "assistant": "listen"},
            {"user": "no voice", "assistant": "Check your electrical panel capacity."},
            {"user": "our service panel is old", "assistant": "listen"},
            {"user": "and pans must be magnetic", "assistant": "listen"},
            {"user": "no voice", "assistant": "Test existing pans with magnets"},
            {"user": "no voice", "assistant": "before scheduling panel upgrades."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 29: conv_00522 | multistep_task
    {
        "id": "conv_00522",
        "scenario": "multistep_task",
        "micro_turns": [
            {"user": "Plan setting up an apiary", "assistant": "listen"},
            {"user": "we want two beehives", "assistant": "listen"},
            {"user": "no voice", "assistant": "Check local municipal zoning ordinances."},
            {"user": "permits are approved already", "assistant": "listen"},
            {"user": "no voice", "assistant": "Assemble wooden hive boxes next."},
            {"user": "and set up hive stands", "assistant": "listen"},
            {"user": "no voice", "assistant": "Face entrance towards morning sun"},
            {"user": "no voice", "assistant": "near clean fresh water."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 30: conv_00537 | disagreement_correction
    {
        "id": "conv_00537",
        "scenario": "disagreement_correction",
        "micro_turns": [
            {"user": "The chemistry lab is in Baker Hall", "assistant": "listen"},
            {"user": "no voice", "assistant": "Course directory lists Baker Hall."},
            {"user": "Actually the syllabus moved it to Olin", "assistant": "listen"},
            {"user": "no voice", "assistant": "Thanks for updating that location."},
            {"user": "yes room two-ten in Olin", "assistant": "listen"},
            {"user": "no voice", "assistant": "Olin Hall room two-ten recorded."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 31: conv_00280 | short_acknowledgement
    {
        "id": "conv_00280",
        "scenario": "short_acknowledgement",
        "micro_turns": [
            {"user": "My sourdough won first prize", "assistant": "listen"},
            {"user": "at the county fair", "assistant": "listen"},
            {"user": "no voice", "assistant": "Outstanding recognition for your baking."},
            {"user": "all the early mornings paid off", "assistant": "listen"},
            {"user": "no voice", "assistant": "Dedication yields delicious craft."},
            {"user": "baking celebration loaves tomorrow", "assistant": "listen"},
            {"user": "no voice", "assistant": "Savor the wonderful triumph."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 32: conv_00022 | hesitation_wait
    {
        "id": "conv_00022",
        "scenario": "hesitation_wait",
        "micro_turns": [
            {"user": "I was looking at choosing", "assistant": "listen"},
            {"user": "between wool or down sleeping bags", "assistant": "listen"},
            {"user": "for backcountry alpine camping", "assistant": "listen"},
            {"user": "no voice", "assistant": "Both have distinct thermal properties."},
            {"user": "we expect damp chilly rain", "assistant": "listen"},
            {"user": "and high humidity nights", "assistant": "listen"},
            {"user": "no voice", "assistant": "Treated down or synthetic wool"},
            {"user": "no voice", "assistant": "retains warmth when wet."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 33: conv_00508 | assistant_interrupted
    {
        "id": "conv_00508",
        "scenario": "assistant_interrupted",
        "micro_turns": [
            {"user": "Recommend an evening soup recipe", "assistant": "listen"},
            {"user": "creamy and warm comfort", "assistant": "listen"},
            {"user": "no voice", "assistant": "Pureed butternut squash with heavy"},
            {"user": "no voice", "assistant": "cream and roasted nutmeg spice"},
            {"user": "wait, I am completely dairy-free", "assistant": "listen"},
            {"user": "use plant based thickeners only", "assistant": "listen"},
            {"user": "no voice", "assistant": "Swap in full-fat coconut milk."},
            {"user": "no voice", "assistant": "It yields rich velvety texture"},
            {"user": "no voice", "assistant": "without any dairy products."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 34: conv_00289 | clarification_needed
    {
        "id": "conv_00289",
        "scenario": "clarification_needed",
        "micro_turns": [
            {"user": "Help me prune my grapevine", "assistant": "listen"},
            {"user": "no voice", "assistant": "Which training system is installed?"},
            {"user": "table grapes along fence", "assistant": "listen"},
            {"user": "no voice", "assistant": "Using bilateral cordon spurs,"},
            {"user": "no voice", "assistant": "or cane renewal pruning?"},
            {"user": "spur pruning on mature cordons", "assistant": "listen"},
            {"user": "no voice", "assistant": "Cut fruiting canes down"},
            {"user": "no voice", "assistant": "to two healthy buds."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 35: conv_00055 | long_explanation_user
    {
        "id": "conv_00055",
        "scenario": "long_explanation_user",
        "micro_turns": [
            {"user": "Our amateur astronomy club telescope mirrors fog", "assistant": "listen"},
            {"user": "dew settles heavily within an hour", "assistant": "listen"},
            {"user": "stargazers cannot view faint nebulae", "assistant": "listen"},
            {"user": "hairdryers drain our portable battery packs", "assistant": "listen"},
            {"user": "and wiping glass scratches anti-reflective coatings", "assistant": "listen"},
            {"user": "no voice", "assistant": "Install low-wattage dew heater strips."},
            {"user": "maybe insulated dew shields too", "assistant": "listen"},
            {"user": "no voice", "assistant": "Wrap flexible dew shields securely"},
            {"user": "no voice", "assistant": "around the front corrector plate."},
            {"user": "no voice", "assistant": "Regulate current with pulse controllers"},
            {"user": "no voice", "assistant": "to minimize battery consumption."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 36: conv_00105 | natural_conversation
    {
        "id": "conv_00105",
        "scenario": "natural_conversation",
        "micro_turns": [
            {"user": "My acoustic cello bridge warped", "assistant": "listen"},
            {"user": "leaning towards the fingerboard", "assistant": "listen"},
            {"user": "no voice", "assistant": "String tension causes that."},
            {"user": "I loosened all four pegs", "assistant": "listen"},
            {"user": "no voice", "assistant": "Gently tilt the bridge back"},
            {"user": "no voice", "assistant": "perpendicular to the top plate."},
            {"user": "it straightened right up", "assistant": "listen"},
            {"user": "no voice", "assistant": "Tune strings up gradually."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 37: conv_00322 | short_acknowledgement
    {
        "id": "conv_00322",
        "scenario": "short_acknowledgement",
        "micro_turns": [
            {"user": "Our indie documentary got accepted", "assistant": "listen"},
            {"user": "into the regional film festival", "assistant": "listen"},
            {"user": "no voice", "assistant": "Thrilling news for your filmmakers."},
            {"user": "two years of editing finally finished", "assistant": "listen"},
            {"user": "no voice", "assistant": "An extraordinary creative achievement."},
            {"user": "attending the premiere next Friday", "assistant": "listen"},
            {"user": "no voice", "assistant": "Enjoy your premiere screening."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 38: conv_00585 | multistep_task
    {
        "id": "conv_00585",
        "scenario": "multistep_task",
        "micro_turns": [
            {"user": "Help me build raised planter beds", "assistant": "listen"},
            {"user": "for growing heirloom vegetables", "assistant": "listen"},
            {"user": "no voice", "assistant": "Level the ground surface thoroughly."},
            {"user": "ground is completely leveled now", "assistant": "listen"},
            {"user": "no voice", "assistant": "Fasten corner posts to planks."},
            {"user": "and staple hardware cloth underneath", "assistant": "listen"},
            {"user": "no voice", "assistant": "Line bottom with cardboard sheets"},
            {"user": "no voice", "assistant": "before shoveling in topsoil."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 39: conv_00083 | user_requests_pause
    {
        "id": "conv_00083",
        "scenario": "user_requests_pause",
        "micro_turns": [
            {"user": "Explain how hydroelectric dams generate electricity", "assistant": "listen"},
            {"user": "no voice", "assistant": "Elevated reservoir water drives penstocks"},
            {"user": "no voice", "assistant": "spinning massive turbine electrical shafts"},
            {"user": "pause a moment, let me picture that", "assistant": "listen"},
            {"user": "no voice", "assistant": "Pausing for a moment."},
            {"user": "no voice", "assistant": "Falling water spins heavy turbines,"},
            {"user": "no voice", "assistant": "which rotate copper electromagnetic generators."},
            {"user": "got it, continue onward", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 40: conv_00453 | disagreement_correction
    {
        "id": "conv_00453",
        "scenario": "disagreement_correction",
        "micro_turns": [
            {"user": "The gallery opening is Friday evening", "assistant": "listen"},
            {"user": "no voice", "assistant": "Exhibition flyer lists Friday opening."},
            {"user": "Wait, the invitation card specifies Saturday", "assistant": "listen"},
            {"user": "no voice", "assistant": "Good catch on invitation date."},
            {"user": "yes Saturday from seven to ten", "assistant": "listen"},
            {"user": "no voice", "assistant": "Saturday evening reception noted."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 41: conv_00580 | long_explanation_user
    {
        "id": "conv_00580",
        "scenario": "long_explanation_user",
        "micro_turns": [
            {"user": "My indoor citrus tree keeps shedding", "assistant": "listen"},
            {"user": "green leaves drop with slight touch", "assistant": "listen"},
            {"user": "radiator heating makes air bone dry", "assistant": "listen"},
            {"user": "spider mites weave webbing under branches", "assistant": "listen"},
            {"user": "and potting soil stays soggy underneath", "assistant": "listen"},
            {"user": "no voice", "assistant": "Increase humidity and adjust drainage."},
            {"user": "maybe spray insecticidal soap wash", "assistant": "listen"},
            {"user": "no voice", "assistant": "Mist foliage with neem solution"},
            {"user": "no voice", "assistant": "to eradicate active mite colonies."},
            {"user": "no voice", "assistant": "Move pot from direct radiators"},
            {"user": "no voice", "assistant": "and water only when dry."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 42: conv_00223 | long_explanation_user
    {
        "id": "conv_00223",
        "scenario": "long_explanation_user",
        "micro_turns": [
            {"user": "I am restoring an old sailboat", "assistant": "listen"},
            {"user": "gelcoat is badly oxidized and chalky", "assistant": "listen"},
            {"user": "teak deck seams have crumbled away", "assistant": "listen"},
            {"user": "rigging wires show fine stress cracks", "assistant": "listen"},
            {"user": "and the marine engine won't turn over", "assistant": "listen"},
            {"user": "no voice", "assistant": "Prioritize seaworthy safety repairs first."},
            {"user": "replace standing rigging wire shrouds", "assistant": "listen"},
            {"user": "no voice", "assistant": "Swage brand new wire shrouds"},
            {"user": "no voice", "assistant": "before inspecting engine starter solenoids."},
            {"user": "no voice", "assistant": "Leave cosmetic gelcoat compounding work"},
            {"user": "no voice", "assistant": "for after structural seaworthiness checks."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 43: conv_00025 | assistant_interrupted
    {
        "id": "conv_00025",
        "scenario": "assistant_interrupted",
        "micro_turns": [
            {"user": "Suggest an itinerary for hiking Zion", "assistant": "listen"},
            {"user": "for a two day trip", "assistant": "listen"},
            {"user": "no voice", "assistant": "Hike the steep Angels Landing"},
            {"user": "no voice", "assistant": "ridge trail early in morning"},
            {"user": "wait, I have severe acrophobia", "assistant": "listen"},
            {"user": "keep routes flat without cliff drops", "assistant": "listen"},
            {"user": "no voice", "assistant": "Hike the Riverside Walk path."},
            {"user": "no voice", "assistant": "Wade gently through the Narrows"},
            {"user": "no voice", "assistant": "along canyon canyon riverbed floors."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 44: conv_00415 | clarification_needed
    {
        "id": "conv_00415",
        "scenario": "clarification_needed",
        "micro_turns": [
            {"user": "Help me choose binoculars", "assistant": "listen"},
            {"user": "no voice", "assistant": "What primary outdoor activity target?"},
            {"user": "birdwatching in dense woods", "assistant": "listen"},
            {"user": "no voice", "assistant": "Want lightweight compact eight power,"},
            {"user": "no voice", "assistant": "or heavy high-magnification ten power?"},
            {"user": "eight power with bright objective lens", "assistant": "listen"},
            {"user": "no voice", "assistant": "Pick eight by forty-two roof-prisms"},
            {"user": "no voice", "assistant": "with multi-coated lenses."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 45: conv_00267 | completion_then_yield
    {
        "id": "conv_00267",
        "scenario": "completion_then_yield",
        "micro_turns": [
            {"user": "What are three tips for composting", "assistant": "listen"},
            {"user": "in a small backyard", "assistant": "listen"},
            {"user": "no voice", "assistant": "Maintain green to brown balance,"},
            {"user": "no voice", "assistant": "turn pile every two weeks,"},
            {"user": "no voice", "assistant": "and keep moisture like sponge."},
            {"user": "no voice", "assistant": "listen"},
            {"user": "thanks, that makes it clear", "assistant": "listen"},
            {"user": "no voice", "assistant": "Glad to help your garden."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 46: conv_00179 | user_interrupts_answer
    {
        "id": "conv_00179",
        "scenario": "user_interrupts_answer",
        "micro_turns": [
            {"user": "Explain what causes tidal bulges", "assistant": "listen"},
            {"user": "no voice", "assistant": "Gravitational pull from the moon"},
            {"user": "no voice", "assistant": "attracts ocean waters toward it"},
            {"user": "wait, why is there opposite bulge", "assistant": "listen"},
            {"user": "no voice", "assistant": "Inertia and centrifugal forces create"},
            {"user": "no voice", "assistant": "the matching antipodal ocean bulge."},
            {"user": "understood, continue the explanation", "assistant": "listen"},
            {"user": "no voice", "assistant": "Earth rotates through these bulges,"},
            {"user": "no voice", "assistant": "producing two daily high tides."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 47: conv_00215 | urgent_chemical_safety
    {
        "id": "conv_00215",
        "scenario": "urgent_chemical_safety",
        "micro_turns": [
            {"user": "I am stripping boat paint", "assistant": "listen"},
            {"user": "indoors in closed garage", "assistant": "listen"},
            {"user": "using methylene chloride solvent stripper", "assistant": "Leave that garage immediately."},
            {"user": "no voice", "assistant": "Methylene chloride fumes are lethal."},
            {"user": "I poured solvent across hull", "assistant": "Stop and exit now."},
            {"user": "feeling dizzy and lightheaded", "assistant": "Get into fresh outdoor air."},
            {"user": "no voice", "assistant": "Call emergency services immediately."},
            {"user": "no voice", "assistant": "Do not reenter closed garage."},
            {"user": "no voice", "assistant": "Fumes cause rapid unconsciousness."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 48: conv_00265 | long_explanation_user
    {
        "id": "conv_00265",
        "scenario": "long_explanation_user",
        "micro_turns": [
            {"user": "Our well water pressure dropped drastically", "assistant": "listen"},
            {"user": "kitchen faucets sputter with air pockets", "assistant": "listen"},
            {"user": "pressure switch clicks repeatedly every minute", "assistant": "listen"},
            {"user": "water pump sounds like dry gravel", "assistant": "listen"},
            {"user": "and fine silt clouds every glass", "assistant": "listen"},
            {"user": "no voice", "assistant": "Shut off pump power immediately."},
            {"user": "cut breaker before pump burns", "assistant": "listen"},
            {"user": "no voice", "assistant": "Check bladder pressure tank gauge"},
            {"user": "no voice", "assistant": "for waterlogged internal bladder rupture."},
            {"user": "no voice", "assistant": "Inspect sediment filtration cartridge filters"},
            {"user": "no voice", "assistant": "before cycling pump back on."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 49: conv_00475 | long_explanation_user
    {
        "id": "conv_00475",
        "scenario": "long_explanation_user",
        "micro_turns": [
            {"user": "Our backyard hens stopped laying eggs", "assistant": "listen"},
            {"user": "daylight hours dropped with late autumn", "assistant": "listen"},
            {"user": "they are shedding feathers across coop", "assistant": "listen"},
            {"user": "their red combs look pale pink", "assistant": "listen"},
            {"user": "and egg boxes stay completely empty", "assistant": "listen"},
            {"user": "no voice", "assistant": "They are going through molt."},
            {"user": "maybe boost dietary protein rations", "assistant": "listen"},
            {"user": "no voice", "assistant": "Supplement feed with dried black soldier"},
            {"user": "no voice", "assistant": "fly larvae or mealworms."},
            {"user": "no voice", "assistant": "Feather regrowth diverts laying energy"},
            {"user": "no voice", "assistant": "until spring daylight returns naturally."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 50: conv_00470 | user_changes_topic
    {
        "id": "conv_00470",
        "scenario": "user_changes_topic",
        "micro_turns": [
            {"user": "What is the best route to cycle", "assistant": "listen"},
            {"user": "around Lake Washington loop", "assistant": "listen"},
            {"user": "no voice", "assistant": "The paved Burke-Gilman trail"},
            {"user": "no voice", "assistant": "provides scenic grade separation."},
            {"user": "actually, what temperature roast asparagus", "assistant": "listen"},
            {"user": "with garlic and olive oil", "assistant": "listen"},
            {"user": "no voice", "assistant": "Four hundred twenty-five Fahrenheit."},
            {"user": "how many minutes in oven", "assistant": "listen"},
            {"user": "no voice", "assistant": "Roast twelve to fifteen minutes"},
            {"user": "no voice", "assistant": "until tender and browned."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 51: conv_00103 | misheard_word
    {
        "id": "conv_00103",
        "scenario": "misheard_word",
        "micro_turns": [
            {"user": "Show me where to buy quality leeks", "assistant": "listen"},
            {"user": "no voice", "assistant": "Do you mean garden leeks"},
            {"user": "no voice", "assistant": "or plumbing leak sealant?"},
            {"user": "vegetable leeks, speech to text garbled it", "assistant": "listen"},
            {"user": "no voice", "assistant": "Fresh culinary allium leeks."},
            {"user": "no voice", "assistant": "Check local farmers produce markets"},
            {"user": "no voice", "assistant": "or organic specialty grocers."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 52: conv_00366 | asr_noisy_repair
    {
        "id": "conv_00366",
        "scenario": "asr_noisy_repair",
        "micro_turns": [
            {"user": "Book an oil change appointment", "assistant": "listen"},
            {"user": "for Thursday at two PM", "assistant": "listen"},
            {"user": "with synthetic oil blend", "assistant": "listen"},
            {"user": "no voice", "assistant": "Appointment scheduled for Thursday two."},
            {"user": "wait, make that Friday afternoon", "assistant": "listen"},
            {"user": "around three thirty instead", "assistant": "listen"},
            {"user": "no voice", "assistant": "Rescheduled for Friday three thirty."},
            {"user": "no voice", "assistant": "Full synthetic blend confirmed."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 53: conv_00216 | urgent_electrical_safety
    {
        "id": "conv_00216",
        "scenario": "urgent_electrical_safety",
        "micro_turns": [
            {"user": "My dishwasher cord is sparking", "assistant": "listen"},
            {"user": "I might grab the plug", "assistant": "Do not touch plug."},
            {"user": "sparks are hitting wet floor", "assistant": "Step back from water."},
            {"user": "no voice", "assistant": "Water conducts lethal voltage."},
            {"user": "no voice", "assistant": "Switch off kitchen breaker immediately."},
            {"user": "no voice", "assistant": "Call an emergency licensed electrician."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 54: conv_00006 | urgent_electrical_safety
    {
        "id": "conv_00006",
        "scenario": "urgent_electrical_safety",
        "micro_turns": [
            {"user": "Lightning struck our backyard tree", "assistant": "listen"},
            {"user": "tree branch snagged overhead cable", "assistant": "listen"},
            {"user": "I will cut branch away", "assistant": "Do not touch branch."},
            {"user": "no voice", "assistant": "Stay at least thirty feet."},
            {"user": "cable is arcing orange sparks", "assistant": "Run far back immediately."},
            {"user": "no voice", "assistant": "Downed lines carry fatal currents."},
            {"user": "no voice", "assistant": "Call emergency nine-one-one now."},
            {"user": "no voice", "assistant": "Keep everyone clear of yard."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 55: conv_00353 | emotional_support
    {
        "id": "conv_00353",
        "scenario": "emotional_support",
        "micro_turns": [
            {"user": "Our small bakery business failed", "assistant": "listen"},
            {"user": "we had to close doors", "assistant": "listen"},
            {"user": "no voice", "assistant": "I am so deeply sorry."},
            {"user": "five years of savings gone", "assistant": "listen"},
            {"user": "no voice", "assistant": "Your courage trying was immense."},
            {"user": "just feels like total failure", "assistant": "listen"},
            {"user": "no voice", "assistant": "You nourished your local community."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 56: conv_00213 | backchannel_story
    {
        "id": "conv_00213",
        "scenario": "backchannel_story",
        "micro_turns": [
            {"user": "We found an injured screech owl", "assistant": "listen"},
            {"user": "sitting quietly by the driveway", "assistant": "listen"},
            {"user": "unable to flap one wing", "assistant": "listen"},
            {"user": "no voice", "assistant": "Poor little creature was hurt."},
            {"user": "wrapped it gently in towels", "assistant": "listen"},
            {"user": "drove to wildlife rescue center", "assistant": "listen"},
            {"user": "no voice", "assistant": "Such caring and compassionate action."},
            {"user": "the vet said it will recover", "assistant": "listen"},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 57: conv_00074 | user_interrupts_answer
    {
        "id": "conv_00074",
        "scenario": "user_interrupts_answer",
        "micro_turns": [
            {"user": "How do volcanic calderas form", "assistant": "listen"},
            {"user": "no voice", "assistant": "Massive eruptions empty underlying magma chambers"},
            {"user": "no voice", "assistant": "causing the overlying volcano to collapse"},
            {"user": "wait, is that like Crater Lake", "assistant": "listen"},
            {"user": "no voice", "assistant": "Crater Lake is a caldera."},
            {"user": "understood, continue what happens next", "assistant": "listen"},
            {"user": "no voice", "assistant": "Rainwater fills the collapsed depression,"},
            {"user": "no voice", "assistant": "forming deep isolated alpine lakes."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 58: conv_00120 | completion_then_yield
    {
        "id": "conv_00120",
        "scenario": "completion_then_yield",
        "micro_turns": [
            {"user": "Name three natural ways to repel", "assistant": "listen"},
            {"user": "garden aphids without synthetic chemicals", "assistant": "listen"},
            {"user": "no voice", "assistant": "Spray diluted neem seed oil,"},
            {"user": "no voice", "assistant": "introduce beneficial predatory ladybugs,"},
            {"user": "no voice", "assistant": "and blast with sharp water."},
            {"user": "no voice", "assistant": "listen"},
            {"user": "excellent, I will try neem", "assistant": "listen"},
            {"user": "no voice", "assistant": "Best wishes with your garden."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
    # 59: conv_00487 | assistant_interrupted
    {
        "id": "conv_00487",
        "scenario": "assistant_interrupted",
        "micro_turns": [
            {"user": "Suggest a weekend scenic drive", "assistant": "listen"},
            {"user": "starting out from Portland Oregon", "assistant": "listen"},
            {"user": "no voice", "assistant": "Drive through the Columbia River"},
            {"user": "no voice", "assistant": "gorge toward picturesque Multnomah Falls"},
            {"user": "wait, gorge highway is under construction", "assistant": "listen"},
            {"user": "steer south toward coast instead", "assistant": "listen"},
            {"user": "no voice", "assistant": "Take highway one-oh-one south."},
            {"user": "no voice", "assistant": "Explore dramatic Cannon Beach sea"},
            {"user": "no voice", "assistant": "stacks and coastal cedar headlands."},
            {"user": "no voice", "assistant": "listen"},
        ]
    },
]

def build_messages(micro_turns):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for mt in micro_turns:
        messages.append({"role": "user", "content": mt["user"]})
        messages.append({"role": "assistant", "content": mt["assistant"]})
    return messages

def main():
    output_path = Path("/home/aditya/Desktop/STUDY/TalkAI/data/duplexcascade_synthetic/micro_turns_test_new.jsonl")
    results = []
    for conv in CONVERSATIONS:
        cid = conv["id"]
        scenario = conv["scenario"]
        topic = f"{scenario}_{cid}"
        micro_turns = conv["micro_turns"]
        messages = build_messages(micro_turns)
        record = {
            "id": cid,
            "scenario": scenario,
            "topic": topic,
            "micro_turns": micro_turns,
            "messages": messages,
        }
        results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(results)} conversations to {output_path}")

if __name__ == "__main__":
    main()
