"""Add explicit Live Preview controls to the scene's existing XRI action asset."""
from pathlib import Path
import json,uuid,copy
root=Path(__file__).resolve().parents[1]
path=root/'Unity/Assets/Samples/XR Interaction Toolkit/2.6.4/Starter Assets/XRI Default Input Actions.inputactions'
data=json.loads(path.read_text(encoding='utf-8'))
controls={'{GripButton}':'gripPressed','{Grip}':'grip','{TriggerButton}':'triggerPressed','{Trigger}':'trigger',
          '{Primary2DAxis}':'thumbstick','{Primary2DAxisClick}':'thumbstickClicked',
          'pointerPosition':'devicePosition','pointerRotation':'deviceRotation'}
added=0
for group in data['maps']:
    group['bindings']=[b for b in group['bindings'] if not b['path'].startswith('<PXR_PTController>')]
    existing={(b['action'],b['path']) for b in group['bindings']}
    for binding in list(group['bindings']):
        if not binding['path'].startswith('<XRController>'):continue
        prefix,control=binding['path'].rsplit('/',1)
        new_path=prefix.replace('<XRController>','<PXR_PTController>')+'/'+controls.get(control,control)
        if (binding['action'],new_path) in existing:continue
        extra=copy.deepcopy(binding);extra['path']=new_path;extra['id']=str(uuid.uuid4())
        # Standalone controls must not inherit composite-part flags or names.
        extra['isPartOfComposite']=False;extra['isComposite']=False;extra['name']=''
        group['bindings'].append(extra);existing.add((binding['action'],new_path));added+=1
path.write_text(json.dumps(data,ensure_ascii=False,indent=4)+'\n',encoding='utf-8')
print('PICO explicit bindings added:',added)
