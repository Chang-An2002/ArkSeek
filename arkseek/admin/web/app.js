/* All user-controlled content is inserted as text, never HTML. */
(() => {
  'use strict';
  const content = document.getElementById('content');
  const error = document.getElementById('error');
  const state = {view:'overview', mode:null, q:'', channel:'all', offset:0, selected:null, detail:null, before:null, status:'all', range:'7d', cid:'', day:'', hour:null, band:'', sort:'recent', layout:'timeline', profileState:'all', log:null, generation:0, detailEpoch:0};
  const labels = {overview:['运行概览','查看系统状态与最近请求'],sessions:['会话记录','群聊按会话查看，成员身份分别保留'],users:['用户与记忆','核对个人画像、原始来源和生效状态'],monitor:['请求监测','区分排队、处理与发送，查看包含排队的等待时间'],logs:['运行日志','查看最近日志片段，辅助定位运行问题'],roadmap:['后续规划','已记录后续阶段，本版提供只读查看']};
  const statuses = {queued:'排队中',processing:'处理中',sending:'发送中',sent:'已发送',send_failed:'发送失败',processing_failed:'处理失败 · 错误提示已发送',failed:'失败',interrupted:'被中断',unknown:'状态未确认'};
  const channels = {group:'群聊',private:'私聊',cli:'本机对话'};
  labels.settings=['运行与配置','管理本后台所属进程与下次启动的非密钥配置'];
  labels.knowledge=['知识库维护','管理项目资料副本、版本和L3索引任务'];
  labels.roadmap=['功能与边界','已实现管理能力及保留的后续范围'];
  function el(tag, attrs={}, ...children) {
    const n = document.createElement(tag);
    for(const [k,v] of Object.entries(attrs)) {
      if(k==='class') n.className=v;
      else if(k.startsWith('on')) n.addEventListener(k.slice(2).toLowerCase(),v);
      else if(k==='value') n.value=v;
      else if(k==='disabled') n.disabled=Boolean(v);
      else n.setAttribute(k,v);
    }
    for(const child of children.flat()) if(child!==null && child!==undefined) n.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return n;
  }
  const btn = (text, fn, cls='') => el('button',{type:'button',class:cls,onClick:fn},text);
  const muted = text => el('span',{class:'muted'},text);
  const time = value => value ? new Intl.DateTimeFormat('zh-CN',{timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date(value)) : '未记录';
  const duration = value => typeof value!=='number' || !Number.isFinite(value) || value<0 ? '未记录' : (value/1000).toFixed(1)+' 秒';
  function pill(status) { return el('span',{class:'pill '+(status==='sent'?'ok':(['queued','processing','sending'].includes(status)?'active':(['failed','send_failed','processing_failed','interrupted'].includes(status)?'bad':'')))},statuses[status]||status); }
  async function api(resource, params={}) {
    const url = new URL('/api/'+resource,location.origin);
    for(const [k,v] of Object.entries(params)) if(v!==null && v!==undefined && v!=='') url.searchParams.set(k,v);
    const response=await fetch(url,{cache:'no-store'});
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||'请求失败');
    return data;
  }
  function showError(exc) { error.textContent=exc.message||String(exc); error.hidden=false; }
  function panel(title,...children) { return el('section',{class:'panel'},el('h2',{},title),...children); }
  function stat(title,value,note) { return el('div',{class:'stat'},el('span',{},title),el('strong',{},value),el('span',{},note)); }
  function pager(data) {
    return el('div',{class:'pager'},el('button',{type:'button',disabled:state.offset===0,onClick:()=>{state.offset=Math.max(0,state.offset-50);load();}},'上一页'),muted(`共 ${data.total} 条`),el('button',{type:'button',disabled:state.offset+50>=data.total,onClick:()=>{state.offset+=50;load();}},'下一页'));
  }
  function searchForm(extra=[]) {
    const input=el('input',{type:'search',value:state.q,placeholder:state.view==='logs'?'在最近日志片段中搜索':'搜索内容或昵称',id:'page-search',maxLength:'200'});
    return el('form',{class:'toolbar',onSubmit:event=>{event.preventDefault();state.q=input.value;state.offset=0;state.before=null;load();}},el('div',{class:'field grow'},el('label',{for:'page-search'},'搜索'),input),...extra,el('button',{type:'submit',class:'primary'},'搜索'));
  }
  function selectField(label,id,options,value,fn) {
    const select=el('select',{id,onChange:event=>fn(event.target.value)},options.map(([key,text])=>el('option',{value:key},text)));
    select.value=value;
    return el('div',{class:'field'},el('label',{for:id},label),select);
  }
  function go(view,selected=null) {
    closeDetail();
    Object.assign(state,{view,selected,q:'',offset:0,before:null,detail:null,generation:state.generation+1});
    location.hash=view;load();
  }
  async function openConversation(id) {state.channel='all';go('sessions',id);}
  async function openUser(id) {state.profileState='all';go('users',id);}
  function factStatus(fact){return fact.management?.mode==='disabled'?'管理员已停用':fact.management?.mode==='override'?(fact.effective?'管理员更正 · 已锁定':'更正值被保护规则屏蔽'):fact.effective?'当前规则允许读取':'已被保护规则屏蔽';}
  function profileCard(user,fact){
    return btn(el('div',{},el('div',{class:'fact-head'},el('strong',{},fact.key),el('span',{class:'pill '+(fact.effective?'ok':'')},factStatus(fact))),el('div',{class:'fact-value'},fact.value),muted('点击核对来源、保护状态与管理历史')),()=>drawer('画像来源 · '+user.name,el('div',{},el('h3',{},fact.key),el('p',{class:'fact-value'},fact.value),el('p',{},factStatus(fact)),fact.management?el('p',{},'原存储值：'+fact.stored_value+' · 管理原因：'+fact.management.reason):null,el('h3',{},'来源原话（原存储）'),el('blockquote',{class:'source-quote'},fact.source_text||'未记录来源'),muted('原存储最后更新 '+time(fact.updated_at)),el('p',{class:'muted'},'来源未绑定唯一消息编号；以下入口打开相关会话。'),...user.conversations.map(con=>el('p',{},btn('查看会话：'+con.name,()=>openConversation(con.id),'link-button'))),ArkManage.profileControls(user,fact))),'item fact-card');
  }
  function requestTable(items) {
    const table=el('table',{},el('thead',{},el('tr',{},['接收时间','会话','状态','排队','总等待','详情'].map(t=>el('th',{},t)))));
    const body=el('tbody');
    for(const r of items) {
      const show=()=>showRequest(r.request_id);
      body.append(el('tr',{class:'request'},el('td',{},time(r.received_at)),el('td',{},btn(r.conversation_name,()=>openConversation(r.conversation_id),'link-button')),el('td',{},pill(r.status)),el('td',{class:'num'},duration(r.queue_ms)),el('td',{class:'num'},duration(r.total_ms)),el('td',{},btn('查看',show,'link-button'))));
    }
    table.append(body);
    return items.length?el('div',{class:'table-wrap'},table):el('p',{class:'empty'},'当前范围没有请求记录');
  }
  function requestDetail(r) {
    const section=panel('请求详情',el('div',{class:'meta'},pill(r.status),muted(time(r.received_at))),el('p',{},'排队 '+duration(r.queue_ms)+' · 接收至结束 '+duration(r.total_ms)),btn('查看该会话',()=>openConversation(r.conversation_id),'link-button'));
    if(r.error_kind)section.append(el('p',{class:'bad-text'},'异常类型：'+r.error_kind));
    if(r.status==='unknown')section.append(el('p',{class:'notice'},(['queued','processing','sending'].includes(r.recorded_status)?'对应进程没有近期心跳，无法确认此请求最终结果。':'原状态未识别，无法确认此请求最终结果。')+'最后记录的阶段：'+(statuses[r.recorded_status]||r.recorded_status)));
    const spans=r.spans.filter(s=>s && Number.isFinite(s.start_ms) && s.start_ms>=0 && Number.isFinite(s.duration_ms) && s.duration_ms>=0);
    if(spans.length) {
      section.append(el('p',{class:'muted'},'以下步骤按处理阶段起点对齐；并行步骤会重叠，不能将耗时相加作为总等待。'));
      const max=Math.max(...spans.map(s=>s.start_ms+s.duration_ms),1);
      for(const s of spans) {
        const mark=el('span',{class:'segment'});mark.style.left=(s.start_ms/max*100)+'%';mark.style.width=Math.max(s.duration_ms/max*100,.5)+'%';
        section.append(el('div',{class:'timeline-row'},el('span',{},s.name+(s.task?' · '+s.task:'')),el('div',{class:'rail',role:'img','aria-label':`${s.name}，开始 ${duration(s.start_ms)}，耗时 ${duration(s.duration_ms)}`},mark),muted(duration(s.duration_ms))));
      }
    } else section.append(el('p',{class:'muted'},'此请求没有分步计时记录。'));
    section.append(el('details',{},el('summary',{},'技术标识'),el('p',{class:'details-id'},r.request_id)));
    return section;
  }
  function closeDetail() {
    state.detail=null;state.detailEpoch++;
    document.getElementById('detail-dialog').close();
  }
  function drawer(title,node) {
    document.getElementById('detail-title').textContent=title;
    document.getElementById('detail-body').replaceChildren(node);
    const d=document.getElementById('detail-dialog');if(!d.open)d.showModal();
  }
  async function showRequest(id,quiet=false) {
    state.detail=id;const epoch=++state.detailEpoch;
    if(!quiet)drawer('请求详情',el('p',{},'正在载入…'));
    try {
      const r=await api('request',{id});
      if(epoch!==state.detailEpoch||state.detail!==id)return;
      drawer('请求详情',el('div',{},requestDetail(r),btn('刷新该请求',()=>showRequest(id,true))));
    }catch(exc){if(epoch===state.detailEpoch)drawer('请求详情',el('p',{class:'bad-text'},exc.message));}
  }
  function drill(options) {
    Object.assign(state,{cid:'',day:'',hour:null,band:'',status:'all',sort:'recent'},options);go('monitor');
  }
  function rangeField() {
    return selectField('时间范围','range',[['today','今天'],['24h','最近24小时'],['7d','最近7天']],state.range,v=>{state.range=v;state.day='';state.hour=null;state.offset=0;load();});
  }
  function coverage(d) {
    const n=el('div',{class:'coverage'},`统计时段 ${time(d.window.start)} — ${time(d.window.end)}（北京时间）`,el('br'),`最早监测记录 ${time(d.recorded_from)}；无记录不代表没有使用。`);
    if(d.capped)n.append(el('p',{class:'bad-text'},`匹配 ${d.matched} 条，仅统计最近10000条；图表与分页均基于这些样本。`));
    const dropped=d.runtimes.reduce((sum,r)=>sum+(r.dropped||0),0);
    if(dropped)n.append(el('p',{class:'bad-text'},`监测队列曾丢弃 ${dropped} 条事件，统计可能不完整。`));
    return n;
  }
  function barList(items,total,select) {
    if(items.some(x=>x.id==='sent')) {
      const failed=items.filter(x=>['failed','send_failed','processing_failed','interrupted'].includes(x.id)).reduce((n,x)=>n+x.count,0);
      items=[items.find(x=>x.id==='sent'),{id:'failed',label:'失败与中断',count:failed},...items.filter(x=>['queued','processing','sending','unknown'].includes(x.id))];
    }
    return el('div',{class:'bar-list'},items.map(x=>{
      const fill=el('span',{class:'bar-fill'});fill.style.width=(total?x.count/total*100:0)+'%';
      return el('button',{type:'button',class:'bar-button',disabled:!x.count,onClick:()=>select(x.id)},fill,el('span',{},x.label),el('strong',{},x.count));
    }));
  }
  function queueTimeline(d) {
    const result=el('div',{},el('p',{class:'muted'},'同一会话按接收时间排列，连续请求共用时间轴；间隔超过5分钟另起一段，每段标明起止时刻。仅展示本页，筛选、分页会隐藏其他请求。点击任一行查看分步记录。'),el('div',{class:'legend'},el('span',{class:'queue-key'},'排队'),el('span',{class:'work-key'},'处理与发送'),el('span',{class:'send-key'},'发送（有独立计时）'),el('span',{class:'unknown-key'},'未记录')));
    const groups=new Map();
    for(const r of d.items){if(!groups.has(r.conversation_id))groups.set(r.conversation_id,[]);groups.get(r.conversation_id).push(r);}
    const ending=r=>{
        const end=Date.parse(r.finished_at);
        return Number.isFinite(end)?end:['queued','processing','sending'].includes(r.status)?Date.parse(d.window.end):Date.parse(r.received_at);
    };
    const clusters=[];
    for(const [cid,all] of groups) {
      all.sort((a,b)=>Date.parse(a.received_at)-Date.parse(b.received_at));
      let chunk=[],lastEnd=-Infinity;
      for(const r of all){if(chunk.length&&Date.parse(r.received_at)-lastEnd>300000){clusters.push([cid,chunk]);chunk=[];}chunk.push(r);lastEnd=Math.max(lastEnd,ending(r));}
      if(chunk.length)clusters.push([cid,chunk]);
    }
    clusters.sort((a,b)=>Math.max(...b[1].map(r=>Date.parse(r.received_at)))-Math.max(...a[1].map(r=>Date.parse(r.received_at))));
    for(const [cid,rows] of clusters) {
      const start=Math.min(...rows.map(r=>Date.parse(r.received_at)));
      const end=Math.max(start+1000,...rows.map(ending)),span=end-start;
      const group=el('section',{class:'queue-group'},el('div',{class:'panel-head'},btn(rows[0].conversation_name,()=>openConversation(cid),'link-button'),muted(`${time(new Date(start).toISOString())} → ${time(new Date(end).toISOString())}`)));
      for(const r of rows) {
        const received=Date.parse(r.received_at),finish=Math.max(received,ending(r));
        const rail=el('div',{class:'queue-rail','aria-hidden':'true'});
        const mark=(a,b,cls)=>{
          const s=el('span',{class:'queue-segment '+cls});s.style.left=((a-start)/span*100)+'%';s.style.width=Math.max((b-a)/span*100,.5)+'%';rail.append(s);
        };
        const known=Number.isFinite(r.queue_ms)&&r.queue_ms>=0;
        const processing=known?Math.min(finish,received+r.queue_ms):received;
        if(finish===received)mark(received,received,'unknown-piece');
        else if(r.status==='unknown'&&!r.finished_at)mark(received,received,'unknown-piece');
        else if(r.status==='queued'&&!r.finished_at)mark(received,finish,'queue-piece');
        else {
          if(known&&processing>received)mark(received,processing,'queue-piece');
          mark(processing,finish,known?'work-piece':'unknown-piece');
          const send=r.spans.find(s=>s&&s.name==='发送'&&Number.isFinite(s.start_ms)&&s.start_ms>=0&&Number.isFinite(s.duration_ms)&&s.duration_ms>=0);
          if(known&&send)mark(Math.min(finish,processing+send.start_ms),Math.min(finish,processing+send.start_ms+send.duration_ms),'send-piece');
        }
        const active=['queued','processing','sending'].includes(r.status)&&!r.finished_at;
        group.append(btn(el('div',{class:'queue-row'},el('div',{},el('strong',{},r.user_name),el('small',{},time(r.received_at))),rail,el('div',{class:'queue-result'},pill(r.status),el('small',{},active?'已等 '+duration(finish-received)+'（未结束）':duration(r.total_ms)))),()=>showRequest(r.request_id),'queue-request'));
      }
      result.append(group);
    }
    if(!groups.size)result.append(el('p',{class:'empty'},'当前筛选没有请求记录'));
    return result;
  }
  async function overview() {
    const d=await api('overview');state.mode=d.mode;
    document.getElementById('mode').textContent=d.mode==='demo'?'演示模式 · 全部为虚构数据':state.managementEnabled?'本机真实数据 · 管理已启用':'本机数据 · 只读接入';
    const nodes=[];
    if(d.mode==='demo')nodes.push(el('div',{class:'notice'},'当前展示虚构样例，便于体验后台布局与操作；没有读取真实聊天和日志。'));
    nodes.push(el('div',{class:'toolbar'},rangeField(),btn('等待最长',()=>drill({sort:'slowest'})),btn('发送失败',()=>drill({status:'send_failed'})),btn('被屏蔽画像',()=>{state.profileState='blocked';go('users');})));
    for(const notice of d.notice)nodes.push(el('p',{class:'notice'},notice));
    if(d.monitor) {
      const a=await api('analytics',{range:state.range}),s=a.summary;
      nodes.push(coverage(a),el('div',{class:'stats'},stat('请求量',s.total,'按接收时间统计'),stat('交互身份',s.users,'同一人的不同群身份分别计数'),stat('总等待中位数',duration(s.median_ms),`${s.completed_samples} 条已结束且有计时记录`),stat('总等待 P90',duration(s.p90_ms),'90%的计时样本不超过此值')));
      const pick=b=>drill({day:b.day,hour:b.hour});
      nodes.push(el('div',{class:'dashboard-grid'},panel('请求量趋势',muted('点击柱形，查看该时段的请求'),ArkCharts.trend(a.series,false,pick)),panel('请求结果',muted('点击状态，筛选对应记录'),barList(Object.entries(a.outcomes).map(([id,count])=>({id,label:statuses[id],count})),s.total,id=>drill({status:id})),el('p',{class:'muted'},`正常发送占比 ${s.sent_rate===null?'—':s.sent_rate+'%'} · 分母为全部记录（含进行中与未知）；不代表回答质量。`)),panel('等待时间趋势',el('div',{class:'legend'},el('span',{class:'total-key'},'总等待中位数'),el('span',{class:'queue-key'},'排队中位数')),ArkCharts.trend(a.series,true,pick),muted('均按接收时段分桶；总等待包含排队，空白点表示没有计时样本。')),panel('总等待分布',barList(a.distribution,s.completed_samples,id=>drill({band:id})),el('p',{class:'muted'},`计时样本 ${s.completed_samples} 条；另有 ${s.missing_duration} 条未结束或未记录耗时。排队中位数 ${duration(s.queue_median_ms)}（${s.queue_samples} 条）。`))));
      nodes.push(panel('会话对比',muted('横条长度表示请求量；点击会话进入请求时间线'),barList(a.groups.map(g=>({id:g.id,label:`${g.name} · 排队中位数 ${duration(g.queue_median_ms)}`,count:g.count})),s.total,id=>drill({cid:id})),!a.groups.length?el('p',{class:'empty'},'当前范围没有会话请求'):null));
      const live=d.monitor.runtimes.some(r=>r.heartbeat_live);
      const runtime=panel('运行状态',el('div',{class:'meta'},el('span',{class:'pill '+(live?'ok':'')},live?'监测进程心跳正常':'未检测到近期进程心跳'),muted('进程心跳（当前）')),
        el('p',{},`所选范围：排队 ${s.queued} · 处理 ${s.processing} · 发送 ${s.sending} · 失败与中断 ${s.failed}`),
        el('p',{class:'muted'},'进程心跳不等于QQ当前连接正常。没有心跳也可能是旧版机器人尚未接入监测。'));
      const dropped=d.monitor.runtimes.reduce((n,r)=>n+(r.dropped||0),0);
      if(dropped)runtime.append(el('p',{class:'bad-text'},`监测队列曾丢弃 ${dropped} 条事件，显示结果可能不完整。`));
      nodes.push(runtime,panel('最近请求',requestTable(a.recent),btn('查看请求时间线',()=>drill({}),'link-button')));
    }
    return nodes;
  }
  async function sessions() {
    const d=await api('sessions',{q:state.q,channel:state.channel,offset:state.offset});
    const list=el('div',{class:'list'});
    for(const s of d.items)list.append(el('button',{type:'button',class:'item '+(state.selected===s.id?'selected':''),onClick:()=>{state.selected=s.id;state.before=null;state.q='';load();}},el('div',{class:'item-title'},el('strong',{},s.name),el('span',{class:'pill'},channels[s.channel])),el('div',{class:'meta'},`${s.messages} 条消息 · ${time(s.last_at)}`),el('div',{class:'preview'},s.preview)));
    if(!d.items.length)list.append(el('p',{class:'empty'},'没有匹配的会话'));
    const detail=el('div');
    if(state.selected) {
      const c=await api('conversation',{id:state.selected,q:state.q,before:state.before});
      const p=panel(c.name,el('p',{class:'muted'},`${channels[c.channel]} · 共 ${c.total} 条消息 · 以下是机器人保存的问答`));
      if(c.channel==='group')p.append(el('p',{class:'muted'},'这里只包含机器人接收到的消息，不是QQ完整群聊。'));
      if(c.summary)p.append(el('details',{},el('summary',{},'查看会话摘要（话题索引）'),el('div',{class:'summary'},c.summary.summary),muted('更新 '+time(c.summary.updated_at)+' · 已合并至消息序号 '+c.summary.summarized_up_to)));
      if(c.has_more)p.append(btn('查看更早记录',()=>{state.before=c.next_before;load();}));
      if(state.before!==null)p.append(btn('返回最新记录',()=>{state.before=null;load();}));
      for(const m of c.items) {
        const name=m.role==='assistant'?'ArkSeek → '+m.speaker_name:m.speaker_name;
        const head=el('div',{class:'message-head'},el('strong',{},m.role==='assistant'?'ArkSeek':m.speaker_name),muted(time(m.created_at)));
        if(m.role==='assistant')head.append(muted('回复对象：'+m.speaker_name));
        if(m.person_id)head.append(btn('查看画像',()=>openUser(m.person_id),'link-button'));
        p.append(el('article',{class:'message','aria-label':name},head,el('div',{class:'message-body'},m.text)));
      }
      if(!c.items.length)p.append(el('p',{class:'empty'},'当前搜索没有匹配的消息'));
      p.append(el('details',{},el('summary',{},'会话技术标识'),el('p',{class:'details-id'},c.id)));
      detail.append(p);
    }else detail.append(panel('选择会话',el('p',{class:'empty'},'从会话列表选择群聊或私聊，查看消息、回复对象和会话摘要。')));
    return [searchForm([selectField('渠道','channel',Object.entries({all:'全部渠道',...channels}),state.channel,value=>{state.channel=value;state.offset=0;load();})]),el('div',{class:'split'},el('div',{},list,pager(d)),detail)];
  }
  async function users() {
    const d=await api('users',{q:state.q,offset:state.offset,profile_state:state.profileState});
    const list=el('div',{class:'list'});
    for(const u of d.items)list.append(el('button',{type:'button',class:'item '+(state.selected===u.id?'selected':''),onClick:()=>{state.selected=u.id;load();}},el('div',{class:'item-title'},el('strong',{},u.name),el('span',{class:'pill'},channels[u.channel])),el('div',{class:'meta'},`${u.scope} · ${u.profile_count} 条画像 · ${u.blocked_count} 条被屏蔽`)));
    const detail=el('div');
    if(state.selected) {
      const u=await api('user',{id:state.selected});
      const p=panel(u.name,el('p',{class:'muted'},'同一个人在不同群或私聊中保留独立身份；本页不自动合并。'));
      for(const fact of u.profiles)p.append(profileCard(u,fact));
      if(!u.profiles.length)p.append(el('p',{class:'empty'},'尚无存储画像'));
      p.append(el('h3',{},'相关会话'));
      for(const c of u.conversations)p.append(el('p',{},btn(c.name,()=>openConversation(c.id),'link-button')));
      p.append(el('details',{},el('summary',{},'身份技术标识'),el('p',{class:'details-id'},u.id)));
      detail.append(p);
    }else detail.append(panel('选择用户',el('p',{class:'empty'},'选择成员，核对画像值、抽取来源和保护规则。')));
    return [searchForm([selectField('画像状态','profile-state',[['all','全部用户'],['blocked','含被屏蔽画像']],state.profileState,v=>{state.profileState=v;state.offset=0;state.selected=null;load();})]),el('div',{class:'split'},el('div',{},list,d.items.length?pager(d):el('p',{class:'empty'},'没有匹配的用户')),detail)];
  }
  async function monitor() {
    const [d,a]=await Promise.all([api('monitor',{status:state.status,range:state.range,conversation_id:state.cid,day:state.day,hour:state.hour,duration_band:state.band,sort:state.sort,offset:state.offset}),api('analytics',{range:state.range})]);
    const change=(key,v)=>{state[key]=v;state.offset=0;load();};
    const groups=a.groups.map(g=>[g.id,g.name]);
    if(state.cid&&!groups.some(g=>g[0]===state.cid))groups.push([state.cid,state.cid]);
    const filters=el('div',{class:'toolbar'},rangeField(),selectField('会话','conversation-filter',[['','全部会话'],...groups],state.cid,v=>change('cid',v)),selectField('请求状态','status',[['all','全部状态'],['failed','失败与中断'],...Object.entries(statuses).filter(([k])=>k!=='failed')],state.status,v=>change('status',v)),selectField('排序','sort',[['recent','最近接收'],['slowest','总等待最长'],['queue','排队最长']],state.sort,v=>change('sort',v)));
    const chips=el('div',{class:'filter-chips'});
    if(state.day)chips.append(btn(`${state.day}${state.hour!==null?' '+state.hour+'时':''} ×`,()=>{state.day='';state.hour=null;state.offset=0;load();}));
    if(state.band)chips.append(btn(`耗时：${{under10:'小于10秒','10to30':'10–30秒','30to60':'30–60秒',over60:'60秒及以上'}[state.band]} ×`,()=>change('band','')));
    if(state.cid||state.day||state.band||state.status!=='all'||state.sort!=='recent')chips.append(btn('清除筛选',()=>drill({})));
    return [filters,chips,coverage(d),el('div',{class:'stats'},stat('筛选记录',d.total,'当前筛选；分页每页50条'),stat('正常发送',d.summary.sent,'会话与时间范围内，未按状态筛选'),stat('失败与中断',d.summary.failed,'会话与时间范围内，未按状态筛选'),stat('等待中位数',duration(d.summary.median_ms),'包含排队；有计时的已结束请求')),
      panel('请求记录',el('div',{class:'toolbar'},btn('时间线',()=>change('layout','timeline'),state.layout==='timeline'?'primary':''),btn('表格',()=>change('layout','table'),state.layout==='table'?'primary':'')),state.layout==='timeline'?queueTimeline(d):requestTable(d.items),pager(d))];
  }
  async function logs() {
    const d=await api('logs');
    if(!state.log && d.items.length)state.log=d.items[0].name;
    const nodes=[searchForm([selectField('日志文件','log-file',d.items.map(f=>[f.name,f.name]),state.log||'',v=>{state.log=v;load();})])];
    if(state.log) {
      const l=await api('log',{name:state.log,q:state.q,limit:100});
      nodes.push(panel('最近日志片段',el('p',{class:'muted'},'最多读取文件末尾256KB、显示500行；搜索仅作用于该片段。常见密钥格式已遮盖，原日志仍按私有数据保存。'),el('pre',{class:'logtext'},l.lines.join('\n')||'没有匹配的日志行'),l.truncated?el('p',{class:'muted'},'部分更早日志未显示。'):null));
    }else nodes.push(panel('运行日志',el('p',{class:'empty'},'尚无控制台日志；机器人启动后会自动保存。')));
    return nodes;
  }
  function roadmap() {
    return [el('div',{class:'roadmap'},panel('已实现 · L1画像管理',el('p',{},'从画像卡片进入更正、停用、解除决定、操作历史和回退。管理记录从启用本功能后开始，不伪造自动抽取历史。停用画像不等于停用原始聊天或历史检索。')),
      panel('已实现 · 运行与配置',el('p',{},'启动/停止本后台持有句柄的进程，维护非密钥配置；显示进程和任务状态。不会接管外部PID。进程存活不代表QQ连接正常。')),
      panel('已实现 · 知识库维护',el('p',{},'纯文本项目副本导入、修改、停用和版本回退；构建/同步L3派生索引。原始科研、Zotero和JC文件只读。')),
      panel('当前边界与后续',el('p',{},'真实模式默认只读，显式启用管理后所有写入仍需影响预览确认。维护需停止机器人/CLI，下次启动生效。历史聊天级停用/恢复、多管理员、远程访问、热更新、实时QQ连接证明和逐条检索证据仍未实现。')))];
  }
  async function load(quiet=false) {
    const generation=++state.generation;
    error.hidden=true;
    const label=labels[state.view]||labels.overview;
    document.getElementById('title').textContent=label[0];document.getElementById('subtitle').textContent=label[1];
    document.querySelectorAll('nav button').forEach(n=>n.setAttribute('aria-current',n.dataset.view===state.view?'page':'false'));
    if(!quiet)content.replaceChildren(el('p',{class:'empty'},'正在载入…'));
    try {
      if(state.managementEnabled===undefined){const access=await ArkManage.status();state.managementEnabled=access.enabled;document.getElementById('access-badge').textContent=access.enabled?(access.demo?'虚构管理 · 可操作':'真实管理 · 需预览'):'数据看板 · 只读';}
      const render={overview,sessions,users,monitor,logs,roadmap,settings:ArkManage.settingsPage,knowledge:ArkManage.knowledgePage}[state.view];
      const nodes=await render();
      if(generation!==state.generation)return;
      content.replaceChildren(...nodes.filter(Boolean));
      document.getElementById('updated').textContent='更新 '+time(new Date().toISOString());
    }catch(exc){if(generation===state.generation){showError(exc);if(!quiet)content.replaceChildren(el('p',{class:'empty'},'当前数据暂不可用；可切换页面或刷新重试。'));}}
  }
  document.querySelectorAll('nav button').forEach(n=>n.addEventListener('click',()=>go(n.dataset.view)));
  document.getElementById('refresh').addEventListener('click',()=>load());
  document.getElementById('close-detail').addEventListener('click',closeDetail);
  document.getElementById('detail-dialog').addEventListener('close',()=>{state.detail=null;state.detailEpoch++;});
  document.getElementById('theme').addEventListener('click',()=>{const current=getComputedStyle(document.documentElement).colorScheme;const next=current==='dark'?'light':'dark';document.documentElement.style.colorScheme=next;try{localStorage.setItem('arkseek-admin-theme',next);}catch(_){};});
  try { const theme=localStorage.getItem('arkseek-admin-theme');if(['dark','light'].includes(theme))document.documentElement.style.colorScheme=theme; }catch(_){}
  setInterval(()=>{if(!document.hidden && document.getElementById('auto').checked && ['overview','monitor'].includes(state.view) && !document.activeElement.matches('input,select,textarea')){if(state.detail)showRequest(state.detail,true);else if(!document.getElementById('detail-dialog').open)load(true);}},10000);
  window.addEventListener('hashchange',()=>{const v=location.hash.slice(1);if(labels[v] && v!==state.view)go(v);});
  const initial=location.hash.slice(1);if(labels[initial])state.view=initial;
  // Obtain the dataset identity even when starting at a bookmarked subpage.
  ArkManage.init({api,el,btn,panel,muted,time,drawer,load,showError});
  api('overview').then(d=>{state.mode=d.mode;document.getElementById('mode').textContent=d.mode==='demo'?'演示模式 · 全部为虚构数据':state.managementEnabled?'本机真实数据 · 管理已启用':'本机数据 · 只读接入';}).catch(showError);
  load();
})();
