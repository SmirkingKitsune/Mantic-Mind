# DeepSeek V4 DSML golden

`encoding/tests/test_input_1.json` and `test_output_1.txt` are the unmodified
case-1 prompt golden from `deepseek-ai/DeepSeek-V4-Pro-0813` at revision
`72e1d3230f6c080a530b0a1d46f8eb4602340597`. The checked-in output has one
repository-final newline; the upstream golden ends directly after its EOS
token, and the test removes that single packaging newline before comparison.

The case exercises system/user/assistant/tool history, two ordered tool
schemas, reasoning history, a DSML invocation, a tool result, Unicode, and the
official Python JSON spacing. `soma_deepseek_v4_codec` requires every encoded
prompt byte to match it.
