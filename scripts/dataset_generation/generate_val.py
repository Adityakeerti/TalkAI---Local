"""
Generates clean, diverse, natural micro-turn dialogues for the 60 validation conversations
in /home/aditya/Desktop/STUDY/TalkAI/data/duplexcascade_synthetic/micro_turns_val_new.jsonl.
Strictly conforms to:
- 600ms micro-turns
- Assistant outputs 'listen' to yield or 3-7 words per micro-turn when speaking
- User outputs 'no voice' during pauses
- Every conversation ends with user: 'no voice', assistant: 'listen'
- High diversity: every conversation has unique topic and distinct entities
- Exact scenario and id matching the validation set
"""

import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a full-duplex conversational assistant. Each input is one 600 ms micro-turn "
    "from streaming speech recognition, or 'no voice' when no speech is detected. "
    "Output exactly 'listen' when you should yield and keep listening; otherwise output "
    "a brief, natural speech fragment that can be spoken now. Do not repeat prior fragments. "
    "Continue an unfinished answer across turns when appropriate. Yield immediately for a "
    "clear interruption or urgent safety risk. Do not treat every pause as the end of the user's thought."
)

# 60 hand-crafted, high-quality, completely distinct dialogues matching each validation record
CONVERSATIONS_DATA = [
    # 0: conv_00570 | backchannel_story
    {
        "id": "conv_00570",
        "scenario": "backchannel_story",
        "turns": [
            ("So I was hiking near Mt Rainier", "listen"),
            ("along the skyline ridge trail", "listen"),
            ("when dense fog rolled in", "listen"),
            ("no voice", "That sounds a bit scary."),
            ("suddenly a mountain goat stepped out", "listen"),
            ("right onto the narrow path", "listen"),
            ("no voice", "Wow, that must have been thrilling!"),
            ("we stood still staring at each other", "listen"),
            ("no voice", "listen"),
        ]
    },
    # 1: conv_00096 | disagreement_correction
    {
        "id": "conv_00096",
        "scenario": "disagreement_correction",
        "turns": [
            ("I thought the flight departs at six", "listen"),
            ("no voice", "My schedule shows six thirty PM."),
            ("Actually the boarding pass says seven fifteen", "listen"),
            ("no voice", "Thanks for correcting that detail."),
            ("yes gate closes in forty minutes", "listen"),
            ("no voice", "Seven fifteen departure, boarding now."),
            ("no voice", "listen"),
        ]
    },
    # 2: conv_00332 | emotional_support
    {
        "id": "conv_00332",
        "scenario": "emotional_support",
        "turns": [
            ("I received some discouraging medical news", "listen"),
            ("and my thoughts keep spiraling today", "listen"),
            ("no voice", "I am truly sorry to hear that."),
            ("I can barely focus on work", "listen"),
            ("no voice", "Be gentle with yourself right now."),
            ("maybe help me draft an absence note", "listen"),
            ("no voice", "Of course. Take a deep breath,"),
            ("no voice", "we will keep it brief and professional."),
            ("no voice", "listen"),
        ]
    },
    # 3: conv_00008 | user_changes_topic
    {
        "id": "conv_00008",
        "scenario": "user_changes_topic",
        "turns": [
            ("What is a good slow cooker meal", "listen"),
            ("with chicken thighs and lentils", "listen"),
            ("no voice", "Chicken lentil stew works great"),
            ("no voice", "with cumin and carrots."),
            ("actually, wait, let me ask about taxes", "listen"),
            ("what is the deadline for filing", "listen"),
            ("no voice", "Federal tax returns are due April fifteenth."),
            ("can I request an automatic extension", "listen"),
            ("no voice", "Yes, form forty-eight sixty-eight"),
            ("no voice", "gives you until mid October."),
            ("no voice", "listen"),
        ]
    },
    # 4: conv_00593 | urgent_chemical_safety
    {
        "id": "conv_00593",
        "scenario": "urgent_chemical_safety",
        "turns": [
            ("I am unclogging the kitchen sink", "listen"),
            ("I poured in sodium hypochlorite", "listen"),
            ("and now adding hydrochloric acid", "Do not mix those!"),
            ("no voice", "That creates deadly chlorine gas!"),
            ("it is sizzling with yellow fumes", "Get away from there now!"),
            ("my throat and eyes are burning", "Evacuate outside immediately!"),
            ("no voice", "Breathe fresh air and call nine-one-one."),
            ("no voice", "Do not re-enter the kitchen."),
            ("no voice", "listen"),
        ]
    },
    # 5: conv_00193 | assistant_interrupted
    {
        "id": "conv_00193",
        "scenario": "assistant_interrupted",
        "turns": [
            ("Can you recommend introductory books on astronomy", "listen"),
            ("for a curious middle schooler", "listen"),
            ("no voice", "Astrophysics for Young People in a Hurry"),
            ("no voice", "by Neil deGrasse Tyson is"),
            ("wait, he is specifically into stargazing", "listen"),
            ("like using binoculars and telescopes", "listen"),
            ("no voice", "Then get Turn Left at Orion."),
            ("no voice", "It has clear sky charts"),
            ("no voice", "perfect for backyard observing."),
            ("no voice", "listen"),
        ]
    },
    # 6: conv_00158 | user_interrupts_answer
    {
        "id": "conv_00158",
        "scenario": "user_interrupts_answer",
        "turns": [
            ("How does a microwave heat food", "listen"),
            ("no voice", "It produces electromagnetic waves that vibrate"),
            ("no voice", "water molecules inside the meal quickly,"),
            ("hold on, why does metal spark inside", "listen"),
            ("no voice", "Metal reflects microwaves, concentrating electrical charges."),
            ("okay, please continue explaining food", "listen"),
            ("no voice", "The vibrating molecules generate rapid friction,"),
            ("no voice", "distributing heat throughout moist ingredients."),
            ("no voice", "listen"),
        ]
    },
    # 7: conv_00227 | emotional_support
    {
        "id": "conv_00227",
        "scenario": "emotional_support",
        "turns": [
            ("My golden retriever passed away this morning", "listen"),
            ("after being sick for three weeks", "listen"),
            ("no voice", "I am deeply sorry for your loss."),
            ("the whole house feels so quiet", "listen"),
            ("no voice", "Losing a loyal pet hurts so much."),
            ("I miss his footsteps on the hardwood", "listen"),
            ("no voice", "Take all the time you need."),
            ("no voice", "He knew how much you loved him."),
            ("no voice", "listen"),
        ]
    },
    # 8: conv_00325 | silence_then_resume
    {
        "id": "conv_00325",
        "scenario": "silence_then_resume",
        "turns": [
            ("I was looking through old family albums", "listen"),
            ("and found my grandfather's journal", "listen"),
            ("no voice", "listen"),
            ("from his journey through Alaska in 1952", "listen"),
            ("with handwritten maps and sketches", "listen"),
            ("no voice", "What an incredible historical treasure!"),
            ("he documented building a log cabin", "listen"),
            ("no voice", "Preserving those stories will mean so much."),
            ("no voice", "listen"),
        ]
    },
    # 9: conv_00571 | assistant_interrupted
    {
        "id": "conv_00571",
        "scenario": "assistant_interrupted",
        "turns": [
            ("How do I improve my running endurance", "listen"),
            ("for an upcoming ten kilometer race", "listen"),
            ("no voice", "Gradually build weekly mileage by"),
            ("no voice", "adding ten percent each week and"),
            ("hold on, my knee starts hurting past 5k", "listen"),
            ("so long runs are causing sharp pain", "listen"),
            ("no voice", "Stop increasing mileage immediately."),
            ("no voice", "Check your footwear and running cadence,"),
            ("no voice", "or consult a physical therapist."),
            ("no voice", "listen"),
        ]
    },
    # 10: conv_00140 | filler_and_mhm
    {
        "id": "conv_00140",
        "scenario": "filler_and_mhm",
        "turns": [
            ("Well uh I wanted some guidance", "listen"),
            ("about asking my manager for a promotion", "listen"),
            ("no voice", "Mm-hm."),
            ("I have exceeded all my sales quarterly targets", "listen"),
            ("but I feel awkward talking about compensation", "listen"),
            ("no voice", "That is very understandable."),
            ("how do I bring up numbers comfortably", "listen"),
            ("no voice", "Frame it around business impact"),
            ("no voice", "and benchmark data from your industry."),
            ("no voice", "listen"),
        ]
    },
    # 11: conv_00563 | emotional_support
    {
        "id": "conv_00563",
        "scenario": "emotional_support",
        "turns": [
            ("I failed my driving road test again", "listen"),
            ("this was my third time trying", "listen"),
            ("no voice", "That must feel really discouraging."),
            ("everyone else makes it look so easy", "listen"),
            ("no voice", "Many excellent drivers needed multiple attempts."),
            ("I just froze during parallel parking", "listen"),
            ("no voice", "You can practice that specific maneuver"),
            ("no voice", "until your muscle memory takes over."),
            ("no voice", "listen"),
        ]
    },
    # 12: conv_00046 | assistant_interrupted
    {
        "id": "conv_00046",
        "scenario": "assistant_interrupted",
        "turns": [
            ("Which city should we visit in Japan", "listen"),
            ("for a week long vacation in autumn", "listen"),
            ("no voice", "Kyoto is famous for vibrant fall foliage,"),
            ("no voice", "ancient temples, and traditional tea houses"),
            ("actually, my partner hates crowded tourist spots", "listen"),
            ("we want somewhere quiet and scenic", "listen"),
            ("no voice", "Consider Kanazawa or Takayama instead."),
            ("no voice", "They offer authentic culture and beauty"),
            ("no voice", "with far fewer tourist crowds."),
            ("no voice", "listen"),
        ]
    },
    # 13: conv_00005 | urgent_chemical_safety
    {
        "id": "conv_00005",
        "scenario": "urgent_chemical_safety",
        "turns": [
            ("Hey assistant, I am cleaning oven racks", "listen"),
            ("I sprayed heavy duty bleach spray", "listen"),
            ("and mixed vinegar to boost power", "Stop, do not combine them!"),
            ("no voice", "Vinegar and bleach produce chloramine gas!"),
            ("it is foaming up with choking vapor", "Get out of the kitchen!"),
            ("coughing really bad right now", "Step outside into open air!"),
            ("no voice", "Call poison control at once."),
            ("no voice", "Do not breathe those fumes."),
            ("no voice", "listen"),
        ]
    },
    # 14: conv_00510 | urgent_electrical_safety
    {
        "id": "conv_00510",
        "scenario": "urgent_electrical_safety",
        "turns": [
            ("A power line fell onto my driveway", "listen"),
            ("it is buzzing and shooting sparks", "Do not go near it!"),
            ("I will nudge it with a broom", "Stay back at least thirty feet!"),
            ("no voice", "Downed power lines can energize surrounding ground."),
            ("no voice", "Stay inside or keep away completely."),
            ("no voice", "Call nine-one-one and the electric utility!"),
            ("no voice", "listen"),
        ]
    },
    # 15: conv_00518 | filler_and_mhm
    {
        "id": "conv_00518",
        "scenario": "filler_and_mhm",
        "turns": [
            ("So like um I am wondering", "listen"),
            ("how to negotiate a commercial lease", "listen"),
            ("no voice", "Mm-hm."),
            ("for a small boutique bakery downtown", "listen"),
            ("the landlord wants a five year commitment", "listen"),
            ("no voice", "That is a major commitment."),
            ("can we ask for tenant improvements", "listen"),
            ("no voice", "Yes, request a tenant improvement allowance"),
            ("no voice", "and a rent-free buildout period."),
            ("no voice", "listen"),
        ]
    },
    # 16: conv_00253 | hesitation_wait
    {
        "id": "conv_00253",
        "scenario": "hesitation_wait",
        "turns": [
            ("I was considering whether to", "listen"),
            ("sell my used hatchback car", "listen"),
            ("or pay for transmission repairs", "listen"),
            ("no voice", "That depends on repair costs."),
            ("the quote came in at three thousand", "listen"),
            ("and the car is worth four", "listen"),
            ("no voice", "Selling might be the wiser financial move"),
            ("no voice", "to avoid ongoing maintenance headaches."),
            ("no voice", "listen"),
        ]
    },
    # 17: conv_00302 | user_changes_topic
    {
        "id": "conv_00302",
        "scenario": "user_changes_topic",
        "turns": [
            ("How do I prune hydrangeas in spring", "listen"),
            ("without cutting off summer blooms", "listen"),
            ("no voice", "Only trim dead old wood,"),
            ("no voice", "as buds form on old stems."),
            ("actually, leave the garden for now", "listen"),
            ("can you proofread my cover letter", "listen"),
            ("no voice", "Gladly. Share the opening paragraph."),
            ("I applied for senior data analyst", "listen"),
            ("no voice", "Highlight your strongest analytical achievements"),
            ("no voice", "and quantify business outcomes directly."),
            ("no voice", "listen"),
        ]
    },
    # 18: conv_00333 | multistep_task
    {
        "id": "conv_00333",
        "scenario": "multistep_task",
        "turns": [
            ("Help me build a bird feeder station", "listen"),
            ("in our back garden this weekend", "listen"),
            ("no voice", "Pick a quiet spot near shrubs."),
            ("I have cedar planks and wire mesh", "listen"),
            ("no voice", "Construct a hopper box with drainage."),
            ("and how do I keep squirrels away", "listen"),
            ("no voice", "Mount a metal baffle underneath"),
            ("no voice", "at least five feet off ground."),
            ("no voice", "listen"),
        ]
    },
    # 19: conv_00328 | long_explanation_user
    {
        "id": "conv_00328",
        "scenario": "long_explanation_user",
        "turns": [
            ("I am redesigning my home pottery studio", "listen"),
            ("because the clay dust settles on everything", "listen"),
            ("my current wheel sits next to storage racks", "listen"),
            ("and there is barely room for wedging tables", "listen"),
            ("plus I need proper wet ventilation", "listen"),
            ("no voice", "Zoning your workspace will help immensely."),
            ("should I separate wet and dry areas", "listen"),
            ("no voice", "Keep damp throwing areas near drains"),
            ("no voice", "and position air filters away from clay."),
            ("no voice", "Wipe surfaces damp rather than sweeping."),
            ("no voice", "listen"),
        ]
    },
    # 20: conv_00273 | natural_conversation
    {
        "id": "conv_00273",
        "scenario": "natural_conversation",
        "turns": [
            ("I attempted baking sourdough bread today", "listen"),
            ("but the loaf turned out gummy inside", "listen"),
            ("no voice", "That is a common baking challenge."),
            ("I might have cut it while warm", "listen"),
            ("no voice", "Let it cool completely first,"),
            ("no voice", "as starches finish setting after baking."),
            ("I will wait two hours next time", "listen"),
            ("no voice", "Your next crumb structure will improve."),
            ("no voice", "listen"),
        ]
    },
    # 21: conv_00031 | silence_then_resume
    {
        "id": "conv_00031",
        "scenario": "silence_then_resume",
        "turns": [
            ("I was cleaning out the attic rafters", "listen"),
            ("and stumbled across an old violin case", "listen"),
            ("no voice", "listen"),
            ("inside was an acoustic instrument from Germany", "listen"),
            ("dated eighteen ninety four in cursive", "listen"),
            ("no voice", "That is a wonderful historical find!"),
            ("the wooden varnish is still in pristine shape", "listen"),
            ("no voice", "Consider showing it to a professional luthier."),
            ("no voice", "listen"),
        ]
    },
    # 22: conv_00573 | urgent_electrical_safety
    {
        "id": "conv_00573",
        "scenario": "urgent_electrical_safety",
        "turns": [
            ("The bathroom light switch is smoking", "listen"),
            ("I see sparks behind the wall plate", "Do not touch the switch!"),
            ("I can spray water to extinguish it", "Never use water on electrical fires!"),
            ("no voice", "Water conducts electricity and causes electrocution."),
            ("no voice", "Shut off the main circuit breaker immediately!"),
            ("no voice", "Use a class C fire extinguisher."),
            ("no voice", "listen"),
        ]
    },
    # 23: conv_00387 | asr_noisy_repair
    {
        "id": "conv_00387",
        "scenario": "asr_noisy_repair",
        "turns": [
            ("Can you reserve a rental SUV", "listen"),
            ("at Denver airport for Thursday morning", "listen"),
            ("pick up around eight AM", "listen"),
            ("no voice", "I can help search rental availability."),
            ("sorry, speech error, I meant Phoenix airport", "listen"),
            ("and pick up around noon instead", "listen"),
            ("no voice", "Updated: Phoenix airport at twelve PM."),
            ("no voice", "Do you prefer standard or four-wheel drive?"),
            ("no voice", "listen"),
        ]
    },
    # 24: conv_00525 | natural_conversation
    {
        "id": "conv_00525",
        "scenario": "natural_conversation",
        "turns": [
            ("My basil plant leaves are drooping", "listen"),
            ("even though the potting soil feels moist", "listen"),
            ("no voice", "Overwatering might be choking the roots."),
            ("the pot has no drainage holes underneath", "listen"),
            ("no voice", "Repot it into a well-draining container"),
            ("no voice", "with plenty of holes at the bottom."),
            ("I have a terracotta pot with saucer", "listen"),
            ("no voice", "Terracotta is perfect for drying soil."),
            ("no voice", "listen"),
        ]
    },
    # 25: conv_00408 | asr_noisy_repair
    {
        "id": "conv_00408",
        "scenario": "asr_noisy_repair",
        "turns": [
            ("Order twelve bottles of sparkling cider", "listen"),
            ("delivered to thirty four Elm Avenue", "listen"),
            ("for the anniversary party on Saturday", "listen"),
            ("no voice", "Got it, twelve bottles to Elm Avenue."),
            ("actually, voice to text glitched, twenty bottles", "listen"),
            ("and the street is Oak Boulevard", "listen"),
            ("no voice", "Corrected: twenty bottles to Oak Boulevard."),
            ("no voice", "Shall I proceed with that delivery address?"),
            ("no voice", "listen"),
        ]
    },
    # 26: conv_00354 | multistep_task
    {
        "id": "conv_00354",
        "scenario": "multistep_task",
        "turns": [
            ("Help me set up an indoor hydroponic garden", "listen"),
            ("in our spare bedroom closet", "listen"),
            ("no voice", "Start with an LED grow light setup."),
            ("I bought a deep water culture reservoir", "listen"),
            ("no voice", "Install the air pump and bubbler stone."),
            ("and what about the nutrient solution mix", "listen"),
            ("no voice", "Balance the water pH around six"),
            ("no voice", "before adding liquid leafy greens fertilizer."),
            ("no voice", "listen"),
        ]
    },
    # 27: conv_00042 | natural_conversation
    {
        "id": "conv_00042",
        "scenario": "natural_conversation",
        "turns": [
            ("Why did my cast iron skillet get sticky", "listen"),
            ("after I tried seasoning it in the oven", "listen"),
            ("no voice", "Too much oil was applied."),
            ("I used a thick coat of flaxseed oil", "listen"),
            ("no voice", "Wipe off almost all oil next time"),
            ("no voice", "before baking at high heat."),
            ("I will scrub it down and restart", "listen"),
            ("no voice", "A paper towel wipe prevents pooling."),
            ("no voice", "listen"),
        ]
    },
    # 28: conv_00176 | user_changes_topic
    {
        "id": "conv_00176",
        "scenario": "user_changes_topic",
        "turns": [
            ("What is the difference between cold brew", "listen"),
            ("and traditional iced drip coffee", "listen"),
            ("no voice", "Cold brew steeps at room temperature"),
            ("no voice", "yielding lower acidity and smoother flavor."),
            ("never mind coffee, my car battery died", "listen"),
            ("which cable clamp connects first during jumps", "listen"),
            ("no voice", "Connect the red positive clamp first."),
            ("to the dead battery terminal right", "listen"),
            ("no voice", "Yes, red to dead positive first,"),
            ("no voice", "then red to good positive terminal."),
            ("no voice", "listen"),
        ]
    },
    # 29: conv_00576 | asr_noisy_repair
    {
        "id": "conv_00576",
        "scenario": "asr_noisy_repair",
        "turns": [
            ("Schedule a dental appointment for cleaning", "listen"),
            ("next Tuesday at nine in the morning", "listen"),
            ("with Doctor Simmons if possible", "listen"),
            ("no voice", "Checking Doctor Simmons on Tuesday morning."),
            ("sorry, microphone misheard, I said Thursday", "listen"),
            ("and with Doctor Patel instead", "listen"),
            ("no voice", "Understood: Doctor Patel on Thursday."),
            ("no voice", "Would nine thirty work for you?"),
            ("no voice", "listen"),
        ]
    },
    # 30: conv_00060 | multistep_task
    {
        "id": "conv_00060",
        "scenario": "multistep_task",
        "turns": [
            ("How do I stain our outdoor wooden deck", "listen"),
            ("that has weathered two harsh winters", "listen"),
            ("no voice", "Power wash and strip old coatings first."),
            ("I already sanded down the rough boards", "listen"),
            ("no voice", "Ensure the timber dries for forty-eight hours."),
            ("which applicator tool works most evenly", "listen"),
            ("no voice", "Use a pad applicator with long strokes,"),
            ("no voice", "working two deck boards at a time."),
            ("no voice", "listen"),
        ]
    },
    # 31: conv_00477 | completion_then_yield
    {
        "id": "conv_00477",
        "scenario": "completion_then_yield",
        "turns": [
            ("What are three key habits for better sleep", "listen"),
            ("when working rotating night shifts", "listen"),
            ("no voice", "Maintain a dark, cool sleeping bedroom,"),
            ("no voice", "wear blue-light blocking glasses heading home,"),
            ("no voice", "and stick to consistent wake hours."),
            ("no voice", "listen"),
            ("thanks, I will buy blackout shades today", "listen"),
            ("no voice", "You are very welcome. Sleep well!"),
            ("no voice", "listen"),
        ]
    },
    # 32: conv_00118 | long_explanation_user
    {
        "id": "conv_00118",
        "scenario": "long_explanation_user",
        "turns": [
            ("I am building an automated pet food dispenser", "listen"),
            ("using an ESP32 microcontroller and servo motor", "listen"),
            ("the 3D printed auger screw keeps jamming", "listen"),
            ("when large kibble pieces get wedged in the chute", "listen"),
            ("causing the motor driver board to overheat", "listen"),
            ("no voice", "Kibble size variation frequently binds rigid augers."),
            ("should I switch to a rotating silicone paddle", "listen"),
            ("no voice", "Flexible silicone fins flex past stubborn kibble"),
            ("no voice", "without stalling your servo drive."),
            ("no voice", "Add current sensing to detect overloads."),
            ("no voice", "listen"),
        ]
    },
    # 33: conv_00506 | question_answer_chunks
    {
        "id": "conv_00506",
        "scenario": "question_answer_chunks",
        "turns": [
            ("How does noise cancelling headphone technology", "listen"),
            ("cancel out loud background engine noise", "listen"),
            ("in simple terms without equations", "listen"),
            ("no voice", "External microphones listen to outside ambient noise."),
            ("no voice", "Internal circuitry calculates the inverted sound wave."),
            ("no voice", "The speaker emits that antiphase wave,"),
            ("no voice", "cancelling the noise before your ear."),
            ("no voice", "listen"),
        ]
    },
    # 34: conv_00559 | long_explanation_user
    {
        "id": "conv_00559",
        "scenario": "long_explanation_user",
        "turns": [
            ("Our community soccer club needs financial help", "listen"),
            ("because field maintenance fees jumped fifty percent", "listen"),
            ("we have seventy kids enrolled across four divisions", "listen"),
            ("many families cannot afford higher registration dues", "listen"),
            ("and our local town grant was cut this year", "listen"),
            ("no voice", "Local business sponsorships can bridge funding gaps."),
            ("what kind of sponsorship perks can we offer", "listen"),
            ("no voice", "Offer company logos printed on team jerseys,"),
            ("no voice", "banners along the field fencing during matches,"),
            ("no voice", "and shout-outs in weekly email newsletters."),
            ("no voice", "listen"),
        ]
    },
    # 35: conv_00527 | question_answer_chunks
    {
        "id": "conv_00527",
        "scenario": "question_answer_chunks",
        "turns": [
            ("Can you explain how geothermal heat pumps", "listen"),
            ("warm houses during subzero freezing weather", "listen"),
            ("when the outdoor air is frigid", "listen"),
            ("no voice", "Underground temperatures remain steady year round."),
            ("no voice", "Fluid circulates through buried ground loops,"),
            ("no voice", "absorbing warmth from the surrounding earth."),
            ("no voice", "A compressor amplifies that heat indoors."),
            ("no voice", "listen"),
        ]
    },
    # 36: conv_00204 | completion_then_yield
    {
        "id": "conv_00204",
        "scenario": "completion_then_yield",
        "turns": [
            ("Give me three tips for learning conversational Spanish", "listen"),
            ("with just fifteen minutes practice a day", "listen"),
            ("no voice", "Label everyday household items with sticky notes,"),
            ("no voice", "listen to Spanish music while commuting,"),
            ("no voice", "and practice speaking short phrases aloud."),
            ("no voice", "listen"),
            ("that sounds manageable, thanks so much", "listen"),
            ("no voice", "¡De nada! Enjoy practicing each day."),
            ("no voice", "listen"),
        ]
    },
    # 37: conv_00201 | disagreement_correction
    {
        "id": "conv_00201",
        "scenario": "disagreement_correction",
        "turns": [
            ("The warranty covers laptop screen replacement right", "listen"),
            ("no voice", "Standard warranty excludes accidental physical cracks."),
            ("Actually I purchased the accidental damage plan", "listen"),
            ("no voice", "My mistake! Accidental coverage handles that."),
            ("yes here is policy number forty eight", "listen"),
            ("no voice", "Policy forty-eight covers total display repair."),
            ("no voice", "listen"),
        ]
    },
    # 38: conv_00077 | filler_and_mhm
    {
        "id": "conv_00077",
        "scenario": "filler_and_mhm",
        "turns": [
            ("Umm I am trying to figure out", "listen"),
            ("which camera lens to buy for portraits", "listen"),
            ("no voice", "Mm-hm."),
            ("I shoot with a full frame Sony body", "listen"),
            ("and love blurred, dreamy background bokeh", "listen"),
            ("no voice", "An eighty-five millimeter prime is ideal."),
            ("should I get the f one point four", "listen"),
            ("no voice", "It provides beautiful depth and separation"),
            ("no voice", "for stunning professional outdoor portrait shots."),
            ("no voice", "listen"),
        ]
    },
    # 39: conv_00124 | misheard_word
    {
        "id": "conv_00124",
        "scenario": "misheard_word",
        "turns": [
            ("Tell me how to cultivate fresh peas", "listen"),
            ("no voice", "Do you mean garden sweet green peas"),
            ("no voice", "or honey producing honey bees?"),
            ("honey bees, sorry speech recognition garbled that", "listen"),
            ("no voice", "Understood: backyard beekeeping essentials."),
            ("no voice", "You will need a Langstroth hive box"),
            ("no voice", "and protective bee smoker and suit."),
            ("no voice", "listen"),
        ]
    },
    # 40: conv_00020 | user_requests_pause
    {
        "id": "conv_00020",
        "scenario": "user_requests_pause",
        "turns": [
            ("Explain the difference between inflation and deflation", "listen"),
            ("no voice", "Inflation is a general price rise,"),
            ("no voice", "reducing currency purchasing power across markets"),
            ("hold on, stop for a second, let me take notes", "listen"),
            ("no voice", "Take your time. Ready whenever you are."),
            ("okay, I wrote that down, please proceed", "listen"),
            ("no voice", "Deflation is when prices broadly decrease,"),
            ("no voice", "which often discourages consumer spending."),
            ("no voice", "listen"),
        ]
    },
    # 41: conv_00003 | backchannel_story
    {
        "id": "conv_00003",
        "scenario": "backchannel_story",
        "turns": [
            ("We sailed our small catamaran off Maui", "listen"),
            ("just as the late afternoon trade winds kicked up", "listen"),
            ("no voice", "That sounds like exciting sailing conditions."),
            ("suddenly a humpback whale breached nearby", "listen"),
            ("less than fifty yards off our bow", "listen"),
            ("no voice", "What an awe-inspiring sight!"),
            ("it slapped its tail three times", "listen"),
            ("no voice", "listen"),
        ]
    },
    # 42: conv_00380 | question_answer_chunks
    {
        "id": "conv_00380",
        "scenario": "question_answer_chunks",
        "turns": [
            ("How does carbon dating determine fossil ages", "listen"),
            ("in straightforward language", "listen"),
            ("without heavy nuclear terminology", "listen"),
            ("no voice", "Living organisms absorb carbon from their environment."),
            ("no voice", "When they die, carbon fourteen decays steadily."),
            ("no voice", "Scientists measure the remaining isotopic ratio,"),
            ("no voice", "calculating elapsed years since death accurately."),
            ("no voice", "listen"),
        ]
    },
    # 43: conv_00551 | urgent_chemical_safety
    {
        "id": "conv_00551",
        "scenario": "urgent_chemical_safety",
        "turns": [
            ("My pool water looks cloudy green", "listen"),
            ("I dumped granular pool chlorine shock", "listen"),
            ("and then added liquid muriatic acid directly", "Get away from the pool!"),
            ("no voice", "Mixing chlorine and acid generates chlorine gas!"),
            ("dense greenish vapor is blowing toward me", "Run upwind into fresh air!"),
            ("I can barely catch my breath", "Call nine-one-one emergency immediately!"),
            ("no voice", "Do not approach the chemical pool area."),
            ("no voice", "listen"),
        ]
    },
    # 44: conv_00336 | natural_conversation
    {
        "id": "conv_00336",
        "scenario": "natural_conversation",
        "turns": [
            ("My homemade pizza dough keeps shrinking back", "listen"),
            ("whenever I try stretching it across the peel", "listen"),
            ("no voice", "The gluten network is too tense."),
            ("I took it straight from the cold fridge", "listen"),
            ("no voice", "Let it rest at room temperature"),
            ("no voice", "for at least one full hour."),
            ("that makes complete sense, thank you", "listen"),
            ("no voice", "Warmer gluten relaxes much more easily."),
            ("no voice", "listen"),
        ]
    },
    # 45: conv_00292 | misheard_word
    {
        "id": "conv_00292",
        "scenario": "misheard_word",
        "turns": [
            ("Where can I purchase a good flour", "listen"),
            ("no voice", "Do you mean baking wheat flour"),
            ("no voice", "or decorative botanical flower arrangements?"),
            ("botanical flowers, speech recognition got confused", "listen"),
            ("no voice", "Understood: fresh botanical flower shops."),
            ("no voice", "Local florists or farmers markets offer"),
            ("no voice", "the freshest seasonal wildflower bouquets."),
            ("no voice", "listen"),
        ]
    },
    # 46: conv_00502 | misheard_word
    {
        "id": "conv_00502",
        "scenario": "misheard_word",
        "turns": [
            ("What is the best way to clean my soul", "listen"),
            ("no voice", "Are you asking about spiritual peace"),
            ("no voice", "or the rubber sole of shoes?"),
            ("shoe sole, auto captions misheard that completely", "listen"),
            ("no voice", "Got it: rubber shoe sole cleaning."),
            ("no voice", "Scrub with baking soda and dish soap,"),
            ("no voice", "using an old soft bristle toothbrush."),
            ("no voice", "listen"),
        ]
    },
    # 47: conv_00152 | urgent_chemical_safety
    {
        "id": "conv_00152",
        "scenario": "urgent_chemical_safety",
        "turns": [
            ("I am stripping paint off vintage furniture", "listen"),
            ("in my small basement workshop", "listen"),
            ("using methylene chloride solvent stripper", "Ventilate the basement immediately!"),
            ("no voice", "Methylene chloride vapor causes lethal carbon monoxide!"),
            ("the fumes are making me dizzy", "Get upstairs into outside air!"),
            ("feeling nauseous and lightheaded", "Evacuate outside and call nine-one-one!"),
            ("no voice", "Never use that stripper indoors."),
            ("no voice", "listen"),
        ]
    },
    # 48: conv_00054 | disagreement_correction
    {
        "id": "conv_00054",
        "scenario": "disagreement_correction",
        "turns": [
            ("Jupiter is the hottest planet in our solar system", "listen"),
            ("no voice", "Jupiter is actually a cold gas giant."),
            ("Oh right, I meant Venus has highest temperatures", "listen"),
            ("no voice", "Correct! Venus has extreme runaway greenhouse heat."),
            ("surface reaches over eight hundred degrees Fahrenheit", "listen"),
            ("no voice", "Hot enough to melt solid lead."),
            ("no voice", "listen"),
        ]
    },
    # 49: conv_00552 | urgent_electrical_safety
    {
        "id": "conv_00552",
        "scenario": "urgent_electrical_safety",
        "turns": [
            ("Our kitchen toaster fell into full dishwater", "listen"),
            ("while still plugged into the live outlet", "Do not reach into the water!"),
            ("I should grab it quickly by the cord", "Do not touch the water or cord!"),
            ("no voice", "You could suffer a fatal electric shock."),
            ("no voice", "Go to the breaker panel immediately!"),
            ("no voice", "Shut off main kitchen power first!"),
            ("no voice", "listen"),
        ]
    },
    # 50: conv_00062 | user_requests_pause
    {
        "id": "conv_00062",
        "scenario": "user_requests_pause",
        "turns": [
            ("Summarize how quantum entanglement works", "listen"),
            ("no voice", "Entangled particles become interconnected, meaning,"),
            ("no voice", "measuring the quantum state of one particle"),
            ("wait, pause right there, phone is ringing", "listen"),
            ("no voice", "No problem. Standing by until you return."),
            ("thanks for waiting, I am back now", "listen"),
            ("no voice", "Instantly reveals the corresponding state"),
            ("no voice", "of the paired partner particle regardless distance."),
            ("no voice", "listen"),
        ]
    },
    # 51: conv_00313 | misheard_word
    {
        "id": "conv_00313",
        "scenario": "misheard_word",
        "turns": [
            ("Where can I adopt a healthy hare", "listen"),
            ("no voice", "Do you mean human hair extensions"),
            ("no voice", "or a rescue pet rabbit hare?"),
            ("rabbit hare, my mic picked up hair", "listen"),
            ("no voice", "Understood: rabbit and hare adoption."),
            ("no voice", "Check the House Rabbit Society shelters"),
            ("no voice", "or your county humane animal shelter."),
            ("no voice", "listen"),
        ]
    },
    # 52: conv_00087 | backchannel_story
    {
        "id": "conv_00087",
        "scenario": "backchannel_story",
        "turns": [
            ("I was camping in Banff National Park", "listen"),
            ("woke up at dawn to clear skies", "listen"),
            ("no voice", "Nothing beats crisp mountain mornings."),
            ("walked to Moraine Lake before any tourists arrived", "listen"),
            ("and the turquoise water was like glass", "listen"),
            ("no voice", "That must have looked like paradise."),
            ("a solitary loon was calling across water", "listen"),
            ("no voice", "listen"),
        ]
    },
    # 53: conv_00277 | assistant_interrupted
    {
        "id": "conv_00277",
        "scenario": "assistant_interrupted",
        "turns": [
            ("What should I pack for hiking in Scotland", "listen"),
            ("during a mid September weekend trek", "listen"),
            ("no voice", "Pack breathable moisture-wicking synthetic base layers,"),
            ("no voice", "sturdy waterproof leather hiking boots, and"),
            ("wait, rain gear is essential right, focus on outerwear", "listen"),
            ("I already packed plenty of base shirts", "listen"),
            ("no voice", "Prioritize a Gore-Tex hooded rain jacket,"),
            ("no voice", "waterproof over-trousers, and dry bag sacks."),
            ("no voice", "listen"),
        ]
    },
    # 54: conv_00568 | hesitation_wait
    {
        "id": "conv_00568",
        "scenario": "hesitation_wait",
        "turns": [
            ("I am trying to decide whether", "listen"),
            ("to adopt a rescue greyhound dog", "listen"),
            ("since I live in a second-floor apartment", "listen"),
            ("no voice", "Greyhounds are surprisingly calm apartment pets."),
            ("I worry about their intense speed", "listen"),
            ("and whether they need huge yards", "listen"),
            ("no voice", "They are mostly gentle couch potatoes"),
            ("no voice", "who only need short daily sprint walks."),
            ("no voice", "listen"),
        ]
    },
    # 55: conv_00189 | natural_conversation
    {
        "id": "conv_00189",
        "scenario": "natural_conversation",
        "turns": [
            ("My acoustic guitar has a persistent fret buzz", "listen"),
            ("on the low E and A strings", "listen"),
            ("no voice", "The truss rod may need adjustment."),
            ("the weather turned cold and very dry recently", "listen"),
            ("no voice", "Dry air causes fretboard shrinkage"),
            ("no voice", "and lowers overall string action."),
            ("I will place a soundhole humidifier inside", "listen"),
            ("no voice", "That usually cures seasonal winter buzz."),
            ("no voice", "listen"),
        ]
    },
    # 56: conv_00162 | completion_then_yield
    {
        "id": "conv_00162",
        "scenario": "completion_then_yield",
        "turns": [
            ("What are three ergonomic tips for laptop users", "listen"),
            ("who work long hours from coffee shops", "listen"),
            ("no voice", "Elevate screen to eye level with stand,"),
            ("no voice", "connect an external compact keyboard and mouse,"),
            ("no voice", "and rest forearms parallel to desk height."),
            ("no voice", "listen"),
            ("great suggestions, ordering a portable laptop stand", "listen"),
            ("no voice", "Your neck and shoulders will thank you!"),
            ("no voice", "listen"),
        ]
    },
    # 57: conv_00007 | short_acknowledgement
    {
        "id": "conv_00007",
        "scenario": "short_acknowledgement",
        "turns": [
            ("I successfully defended my master's thesis today", "listen"),
            ("after months of grueling research analysis", "listen"),
            ("no voice", "Huge congratulations on your achievement!"),
            ("I feel such immense relief right now", "listen"),
            ("no voice", "You worked so hard for this moment."),
            ("time to celebrate with close friends tonight", "listen"),
            ("no voice", "Enjoy every second of your celebration!"),
            ("no voice", "listen"),
        ]
    },
    # 58: conv_00304 | silence_then_resume
    {
        "id": "conv_00304",
        "scenario": "silence_then_resume",
        "turns": [
            ("We were exploring an antique thrift shop", "listen"),
            ("and spotted a dusty wooden music box", "listen"),
            ("no voice", "listen"),
            ("when we wound the brass winding key", "listen"),
            ("it played Debussy's Clair de Lune perfectly", "listen"),
            ("no voice", "What an enchanting little discovery!"),
            ("we ended up buying it for twenty dollars", "listen"),
            ("no voice", "A delightful keepsake with real charm."),
            ("no voice", "listen"),
        ]
    },
    # 59: conv_00232 | hesitation_wait
    {
        "id": "conv_00232",
        "scenario": "hesitation_wait",
        "turns": [
            ("I was hesitating about whether to", "listen"),
            ("switch from gas heating to a heat pump", "listen"),
            ("for our older two story home", "listen"),
            ("no voice", "Insulation quality makes a huge difference."),
            ("our attic has very minimal fiberglass insulation", "listen"),
            ("and original single pane wooden windows", "listen"),
            ("no voice", "Air seal and insulate the attic first"),
            ("no voice", "before sizing your new heat pump system."),
            ("no voice", "listen"),
        ]
    },
]


def build_conversation(item: dict) -> dict:
    cid = item["id"]
    scenario = item["scenario"]
    # Topic matches <scenario>_variant_<id_number>
    id_num = cid.replace("conv_", "")
    topic = f"{scenario}_variant_{id_num}"
    
    micro_turns = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for u, a in item["turns"]:
        micro_turns.append({"user": u, "assistant": a})
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
        
    return {
        "id": cid,
        "scenario": scenario,
        "topic": topic,
        "micro_turns": micro_turns,
        "messages": messages,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    orig_val_path = base_dir / "micro_turns_val.jsonl"
    out_val_path = base_dir / "micro_turns_val_new.jsonl"
    
    # Read original validation to verify IDs and scenarios match
    with open(orig_val_path, "r", encoding="utf-8") as f:
        orig_items = [json.loads(line) for line in f]
        
    assert len(CONVERSATIONS_DATA) == 60, f"Expected 60 conversations, got {len(CONVERSATIONS_DATA)}"
    assert len(orig_items) == 60, f"Expected 60 orig items, got {len(orig_items)}"
    
    # Validate each conversation
    seen_prompts = set()
    output_records = []
    
    for idx, (data, orig) in enumerate(zip(CONVERSATIONS_DATA, orig_items)):
        assert data["id"] == orig["id"], f"ID mismatch at {idx}: {data['id']} != {orig['id']}"
        assert data["scenario"] == orig["scenario"], (
            f"Scenario mismatch at {idx}: {data['scenario']} != {orig['scenario']}"
        )
        
        # Verify turns rules
        last_turn = data["turns"][-1]
        assert last_turn == ("no voice", "listen"), f"{data['id']} does not end with ('no voice', 'listen')"
        
        # Check first non-empty user turns for uniqueness
        first_user = next(u for u, _ in data["turns"] if u != "no voice")
        assert first_user.lower() not in seen_prompts, f"Duplicate user question found: {first_user}"
        seen_prompts.add(first_user.lower())
        
        # Check word counts
        for u, a in data["turns"]:
            if a != "listen":
                wcount = len(a.split())
                # 3-7 words rule (or slight variation up to 8 max allowed)
                if data["scenario"] == "filler_and_mhm" and a == "Mm-hm.":
                    pass
                else:
                    assert 3 <= wcount <= 7, f"Word count out of range ({wcount}): '{a}' in {data['id']}"
                
        conv_obj = build_conversation(data)
        output_records.append(conv_obj)
        
    with open(out_val_path, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {len(output_records)} validation dialogues in {out_val_path}")


if __name__ == "__main__":
    main()
