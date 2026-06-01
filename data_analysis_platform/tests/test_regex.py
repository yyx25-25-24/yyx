import re
s='{"summary": "测试通过", "top5": [{"id":1}]}'
print('s=', s)
m=re.search(r'(\{[\s\S]*?\}|\[[\s\S]*?\])', s)
print('match=', m)
if m:
    print('group=', m.group(0))
else:
    print('no match')
