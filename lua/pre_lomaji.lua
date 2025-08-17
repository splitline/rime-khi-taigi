local F = {}

function F.init(env)
  local config = env.engine.schema.config
  local format = config:get_list('translator/comment_format')
  local preedit = config:get_list('translator/preedit_format')
  F.format = Projection()
  F.format:load(format)
  F.preedit = Projection()
  F.preedit:load(preedit)
end

function F.func( input, env )
  local ctx = env.engine.context
  local is_lomaji = ctx:get_option("lomaji")

  local seg = ctx.composition:back()
  local _start, _end = seg._start, seg._end
  local text = ctx.input:sub(_start+1)

  local lomaji_text = F.format:apply(text, true)
  if is_lomaji and 
      text:find("%d") and
      lomaji_text and #lomaji_text > 0 then
    local lomaji_cand = Candidate("Lomaji", _start, _end, lomaji_text, "")
    lomaji_cand.preedit = F.preedit:apply(text, true)
    yield(lomaji_cand)
  end
  for cand in input:iter() do
    yield(cand)
  end
end

return F